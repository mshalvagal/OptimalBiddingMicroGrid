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

actions2=zeros(NEpisodes,NBlocks);

% agent_params.alpha=agent_params.alpha*1.1;
% agent_params.epsilon=agent_params.epsilon*1.1;

bat_storage=zeros(NEpisodes,NBlocks);
bat_storage_o=zeros(NEpisodes,NBlocks);
bat_storage_op=zeros(NEpisodes,NBlocks);
bat_storage2=zeros(NEpisodes,NBlocks);
num_charge_cycles = 0;
num_charge_cycles_o = 0;
num_charge_cycles_op = 0;
num_charge_cycles2 = 0;

performance_measures_agent = struct('reward',zeros(NEpisodes,NBlocks),'wastage',zeros(NEpisodes,NBlocks),'bat_charge_cost',zeros(NEpisodes,NBlocks),'cost',zeros(NEpisodes,NBlocks));
performance_measures_oracle = performance_measures_agent;
performance_measures_oracle2 = performance_measures_agent;
performance_measures_agent2 = performance_measures_agent;

for k=1:NEpisodes
%     agent_params.alpha=agent_params.alpha/1.005;
%     agent_params.epsilon=agent_params.epsilon/1.005;
    
%     bat_eff = max(bat_eff_final,bat_eff_init - (bat_eff_init-bat_eff_final)*num_charge_cycles/bat_eff_lifetime);
%     bat_eff_o = max(bat_eff_final,bat_eff_init - (bat_eff_init-bat_eff_final)*num_charge_cycles_o/bat_eff_lifetime);
%     bat_eff_op = max(bat_eff_final,bat_eff_init - (bat_eff_init-bat_eff_final)*num_charge_cycles_op/bat_eff_lifetime);
%     bat_eff2 = max(bat_eff_final,bat_eff_init - (bat_eff_init-bat_eff_final)*num_charge_cycles2/bat_eff_lifetime);
    bat_eff = bat_eff_init;
    bat_eff_o = bat_eff_init;
    bat_eff_op = bat_eff_init;
    bat_eff2 = bat_eff_init;
    
    i=ceil(rand()*NDays)+30;        % Leaving the first 30 days out of the sample experience
    bat_soc_init = bat_charge_min + rand()*(bat_cap-bat_charge_min);
    
    %% Agent sim
    actions(k,:) = agent_sim(i,bat_soc_init,env_params,agent_params,energy_data);
    actions2(k,:) = agent_sim2(i,bat_soc_init,env_params,agent_params,energy_data,tetha);
    
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
    agent_params.weights = ann_train(state,agent_params.weights,reward,q_sa,qnext_sa,gamma);

    td_error(k) = sum(abs(reward + gamma*qnext_sa - q_sa),1);
    
    if rem(k,update_freq) == 0
        agent_params.target_weights = agent_params.weights;
    end
     
    %% Actual energy usage for linear agent
    [performance,bat_storage2(k,:)] = evaluate_actions(i,bat_soc_init,bat_eff2,env_params,energy_data,actions2(k,:));
    
    performance_measures_agent2.reward(k,:) = performance.reward2;
    performance_measures_agent2.wastage(k,:) = performance.wastage;
    performance_measures_agent2.bat_charge_cost(k,:) = performance.bat_charge_cost;
    performance_measures_agent2.cost(k,:) = performance.actual_cost;
    num_charge_cycles2 = num_charge_cycles2 + performance.charge_cycles;
    
    %% Train Linear Agent
    reward = performance_measures_agent2.reward(k,:)';
    
    for j = 1:NBlocks
        if j~=NBlocks
            [~,qnext_sa(j)] = greedy(demand_norm_a(i,j+1),solar_norm_a(i,j+1),bat_storage2(k,j+1)/bat_cap,min_bid_q,max_bid_q, acp_a(i,j+1)/max_acp, (j+1)/NBlocks,tetha);
        end
        present_state = basisExpansion(demand_norm_a(i,j), solar_norm_a(i,j), bat_storage2(k,j)/bat_cap, actions2(k,j)/max_bid_q, acp_a(i,j)/max_acp, j/NBlocks);
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
end

%% Plotting Performance Curves
totalWaste = 0;
totalWaste_o = 0;
totalWaste_op = 0;
totalWaste2 = 0;
for i=1:NEpisodes
    totalWaste = totalWaste + sum(performance_measures_agent.wastage(i, :));
    wastage_avg(i) = totalWaste / i;
    totalWaste_o = totalWaste_o + sum(performance_measures_oracle.wastage(i, :));
    wastage_avg_o(i) = totalWaste_o / i;
    totalWaste_op = totalWaste_op + sum(performance_measures_oracle2.wastage(i, :));
    wastage_avg_op(i) = totalWaste_op / i;
    totalWaste2 = totalWaste2 + sum(performance_measures_agent2.wastage(i, :));
    wastage_avg2(i) = totalWaste2 / i;
end

figure()
plot(1:NEpisodes, wastage_avg);
hold on;
plot(1:NEpisodes, wastage_avg_o);
plot(1:NEpisodes, wastage_avg_op);
xlabel('Episodes','FontSize', 14);
plot(1:NEpisodes, wastage_avg2);
ylabel('Averaged daily wastage','FontSize', 14);
legend('Agent','Oracle','Oracle2','Linear Agent');

totalbattBen = 0;
totalbattBen_o = 0;
totalbattBen_op = 0;
totalbattBen2 = 0;
for i=1:NEpisodes
    totalbattBen = totalbattBen + sum(performance_measures_agent.bat_charge_cost(i, :));
    bat_benefit_avg(i) = totalbattBen / i;
    totalbattBen_o = totalbattBen_o + sum(performance_measures_oracle.bat_charge_cost(i, :));
    bat_benefit_avg_o(i) = totalbattBen_o / i;
    totalbattBen_op = totalbattBen_op + sum(performance_measures_oracle2.bat_charge_cost(i, :));
    bat_benefit_avg_op(i) = totalbattBen_op / i;
    totalbattBen2 = totalbattBen2 + sum(performance_measures_agent2.bat_charge_cost(i, :));
    bat_benefit_avg2(i) = totalbattBen2 / i;
end

figure()
plot(1:NEpisodes, bat_benefit_avg);
% plot(1:NEpisodes, mean(bat_charge_cost,2));
hold on;
plot(1:NEpisodes, bat_benefit_avg_o);
plot(1:NEpisodes, bat_benefit_avg_op);
plot(1:NEpisodes, bat_benefit_avg2);
% plot(1:NEpisodes, mean(bat_charge_cost_o,2));
xlabel('Episodes','FontSize', 12);
ylabel('Averaged Savings in the bill due to battery','FontSize', 12);
legend('Agent','Oracle','Oracle2','Linear Agent');

for i= 1:NBlocks
    sum_0 = 0;
    sum_1 = 0;
    sum_o = 0;
    sum_op = 0;
    sum_2 = 0;
    for j = 1:NEpisodes
        sum_0 = sum_0 + cost_without_battery(j, i);
        bill_without_batt(j, i) = sum_0 / j;
        sum_1 = sum_1 + performance_measures_agent.cost(j, i);
        actual_bill(j, i) = sum_1 / j;
        sum_o = sum_o + performance_measures_oracle.cost(j, i);
        oracle_bill(j, i) = sum_o / j;
        sum_op = sum_op + performance_measures_oracle2.cost(j, i);
        oracle_pred_bill(j, i) = sum_op / j;
        sum_2 = sum_2 + performance_measures_agent2.cost(j, i);
        actual_bill2(j, i) = sum_2 / j;
    end
end

figure()
plot(1:NEpisodes,sum(actual_bill,2))
hold on
plot(1:NEpisodes,sum(oracle_bill,2))
plot(1:NEpisodes,sum(oracle_pred_bill,2))
plot(1:NEpisodes,sum(actual_bill2,2))
plot(1:NEpisodes,sum(bill_without_batt,2))
xlabel('Episodes','FontSize', 12);
ylabel('Daily bill','FontSize', 12);
legend('Actual bill','Oracle bill','Oracle 2 bill','Linear Agent bill','Bill without battery');

figure()
plot(td_error);

Error = mean(mean((cost_without_battery-performance_measures_agent.cost),1))
Error = mean(mean((cost_without_battery-performance_measures_agent2.cost),1))