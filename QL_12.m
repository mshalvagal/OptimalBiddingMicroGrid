% Gradient Q Learning implementation for electricity market bidding
% clc
clear
% rng('default');

%% Initilizing variables
init_env_params

%% Setting things up
qnext_sa=zeros(NBlocks,1);
q_sa=zeros(NBlocks,1);
actions=zeros(NEpisodes,NBlocks);
td_error = zeros(NEpisodes,1);
oracle_actions=zeros(NEpisodes,NBlocks);

actions_la=zeros(NEpisodes,NBlocks);

% agent_params.alpha=agent_params.alpha*1.1;
% agent_params.epsilon=agent_params.epsilon*1.1;

bat_storage=zeros(NEpisodes,NBlocks);
bat_storage_o=zeros(NEpisodes,NBlocks);
bat_storage_op=zeros(NEpisodes,NBlocks);
bat_storage_la=zeros(NEpisodes,NBlocks);
num_charge_cycles = 0;
num_charge_cycles_o = 0;
num_charge_cycles_op = 0;
num_charge_cycles_la = 0;

performance_measures_agent = struct('reward',zeros(NEpisodes,NBlocks),'wastage',zeros(NEpisodes,NBlocks),'bat_charge_cost',zeros(NEpisodes,NBlocks),'cost',zeros(NEpisodes,NBlocks));
performance_measures_oracle = performance_measures_agent;
performance_measures_oracle2 = performance_measures_agent;
performance_measures_agent_la = performance_measures_agent;

% test_idx = ceil(rand()*NDays)+30;
test_idx = 100;
test_cost = [];
test_reward = [];
test_cost_la = [];

for k=1:NEpisodes
%     agent_params.alpha=agent_params.alpha/1.005;
%     agent_params.epsilon=agent_params.epsilon/1.005;
    
%     bat_eff = max(bat_eff_final,bat_eff_init - (bat_eff_init-bat_eff_final)*num_charge_cycles/bat_eff_lifetime);
%     bat_eff_o = max(bat_eff_final,bat_eff_init - (bat_eff_init-bat_eff_final)*num_charge_cycles_o/bat_eff_lifetime);
%     bat_eff_op = max(bat_eff_final,bat_eff_init - (bat_eff_init-bat_eff_final)*num_charge_cycles_op/bat_eff_lifetime);
%     bat_eff_la = max(bat_eff_final,bat_eff_init - (bat_eff_init-bat_eff_final)*num_charge_cycles_la/bat_eff_lifetime);
    bat_eff = bat_eff_init;
    bat_eff_o = bat_eff_init;
    bat_eff_op = bat_eff_init;
    bat_eff_la = bat_eff_init;
    
    i=ceil(rand()*NDays)+30;        % Leaving the first 30 days out of the sample experience
    i = 100;
%    bat_soc_init = bat_charge_min + rand()*(bat_cap-bat_charge_min);
    bat_soc_init = 0.5*bat_cap;
    
    %% Agent sim
    actions(k,:) = agent_sim(i,bat_soc_init,env_params,agent_params,energy_data);
    actions_la(k,:) = agent_sim2(i,bat_soc_init,env_params,agent_params,energy_data,tetha);
    
    %% Oracle sim
    bid_q_o = oracle_sim(i,bat_soc_init,'actual',env_params,energy_data);
    oracle_actions(k,:) = bid_q_o;
%     bid_q_o = 5+(30-5).*rand(1,24);

    %% Oracle sim
    bid_q_op = oracle_sim(i,bat_soc_init,'predicted',env_params,energy_data);
    
    %% Actual energy usage for agent
    [performance,bat_storage(k,:)] = evaluate_actions(i,bat_soc_init,bat_eff,env_params,energy_data,actions(k,:));
    
    performance_measures_agent.reward(k,:) = performance.reward;
    performance_measures_agent.wastage(k,:) = performance.wastage;
    performance_measures_agent.bat_charge_cost(k,:) = performance.bat_charge_cost;
    performance_measures_agent.cost(k,:) = performance.actual_cost;
    num_charge_cycles = num_charge_cycles + performance.charge_cycles;
    
    %% Train Agent
    state = [demand_norm_a(i,:); solar_norm_a(i,:); bat_storage(k,:)/bat_cap; acp_a(i,:)/max_acp; (1:NBlocks)/NBlocks; actions(k,:)/max_bid_q]';
    [~,qnext_sa(1:end-1)] = greedy_nn(state(2:end,1:end-1),agent_params.target_weights,env_params);
    reward = performance_measures_agent.reward(k,:)';
    
    q_sa = ann_pred(state,agent_params.weights);
    agent_params = ann_train(state,agent_params,reward,q_sa,qnext_sa,gamma);

    td_error(k) = sum(abs(reward + gamma*qnext_sa - q_sa),1);
    
    if rem(k,update_freq) == 0
        agent_params.target_weights = agent_params.weights;
        if agent_params.epsilon > 0
        agent_params.epsilon = max(0,agent_params.epsilon - 0.02);
        end
    end
     
    %% Actual energy usage for linear agent
    [performance,bat_storage_la(k,:)] = evaluate_actions(i,bat_soc_init,bat_eff_la,env_params,energy_data,actions_la(k,:));
    
    performance_measures_agent_la.reward(k,:) = performance.reward2;
    performance_measures_agent_la.wastage(k,:) = performance.wastage;
    performance_measures_agent_la.bat_charge_cost(k,:) = performance.bat_charge_cost;
    performance_measures_agent_la.cost(k,:) = performance.actual_cost;
    num_charge_cycles_la = num_charge_cycles_la + performance.charge_cycles;
    
    %% Train Linear Agent
    reward = performance_measures_agent_la.reward(k,:)';
    
    for j = 1:NBlocks
        if j~=NBlocks
            [~,qnext_sa(j)] = greedy(demand_norm_a(i,j+1),solar_norm_a(i,j+1),bat_storage_la(k,j+1)/bat_cap,min_bid_q,max_bid_q, acp_a(i,j+1)/max_acp, (j+1)/NBlocks,tetha);
        end
        present_state = basisExpansion(demand_norm_a(i,j), solar_norm_a(i,j), bat_storage_la(k,j)/bat_cap, actions_la(k,j)/max_bid_q, acp_a(i,j)/max_acp, j/NBlocks);
        q_sa(j)=valuefn(present_state,tetha);
        tetha = tetha + alpha*(reward(j) + gamma*qnext_sa(j) - q_sa(j))*present_state';
    end
    
    %% If there was no battery
	netdemand = (demand_a(i,:)-solar_a(i,:));
    idx = find(acp_a(i,:)<grid_rate);
    cost_without_battery(k,idx) = netdemand(idx).*acp_a(i,idx);
    idx = find(acp_a(i,:)>=grid_rate);
    cost_without_battery(k,idx) = netdemand(idx).*grid_rate;
%     cost_without_battery(k,:) = netdemand*grid_rate;
    
    %% Actual energy usage for oracle
    [performance,bat_storage_o(k,:)] = evaluate_actions(i,bat_soc_init,bat_eff_o,env_params,energy_data,bid_q_o);
    
    performance_measures_oracle.reward(k,:) = performance.reward;
    performance_measures_oracle.wastage(k,:) = performance.wastage;
    performance_measures_oracle.bat_charge_cost(k,:) = performance.bat_charge_cost;
    performance_measures_oracle.cost(k,:) = performance.actual_cost;
    num_charge_cycles_o = num_charge_cycles_o + performance.charge_cycles;
    
    %% Actual energy usage for oracle 2
    [performance,bat_storage_op(k,:)] = evaluate_actions(i,bat_soc_init,bat_eff_op,env_params,energy_data,bid_q_op);
    
    performance_measures_oracle2.reward(k,:) = performance.reward;
    performance_measures_oracle2.wastage(k,:) = performance.wastage;
    performance_measures_oracle2.bat_charge_cost(k,:) = performance.bat_charge_cost;
    performance_measures_oracle2.cost(k,:) = performance.actual_cost;
    num_charge_cycles_op = num_charge_cycles_op + performance.charge_cycles;
    
    if rem(k,100) == 0
        test_actions = agent_sim(test_idx,bat_soc_init,env_params,agent_params,energy_data);
        [performance,~] = evaluate_actions(test_idx,bat_soc_init,bat_eff,env_params,energy_data,test_actions);
        test_cost = [test_cost; sum(performance.actual_cost,2)];
        test_reward = [test_reward; sum(performance.reward,2)];
                
        test_actions = agent_sim2(test_idx,bat_soc_init,env_params,agent_params,energy_data,tetha);
        [performance,~] = evaluate_actions(test_idx,bat_soc_init,bat_eff,env_params,energy_data,test_actions);
        test_cost_la = [test_cost_la; sum(performance.actual_cost,2)];
    end
end

%% Plotting Performance Curves
% plot_performance_curves

figure()
plot(td_error);

figure()
plot(test_cost);
hold on;
plot(test_cost_la);
legend('nn','linear')

figure()
plot(test_reward);

Error = mean(mean((cost_without_battery-performance_measures_agent.cost),1))
Error = mean(mean((cost_without_battery-performance_measures_agent_la.cost),1))