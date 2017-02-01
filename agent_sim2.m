function actions = agent_sim2(i,bat_soc,env_params,agent_params,energy_data,tetha)

    NBlocks = env_params.NBlocks;
    grid_pred=zeros(1,NBlocks);
    actions=zeros(1,NBlocks);
    
    bat_cap = env_params.bat_params.bat_cap;
    bat_charge_rate = env_params.bat_params.bat_charge_rate;
    bat_charge_min = env_params.bat_params.bat_charge_min;
    
    demand_pred = energy_data.demand_pred(i,:);
    solar_pred = energy_data.solar_pred(i,:);

    for j = 1:NBlocks
        mean_acp = mean(energy_data.acp_a(i-30:i-1,j));
        state = [energy_data.demand_norm(i,j),energy_data.solar_norm(i,j),bat_soc/bat_cap,mean_acp/energy_data.max_acp,j/NBlocks];
        [bid_q,~] = egreedy(demand_pred(j),solar_pred(j),bat_soc,env_params.min_bid_q,env_params.max_bid_q,env_params.bid_p,j,agent_params.epsilon,tetha);
        actions(j)=bid_q;
        % Next State based on predictions
        if bat_soc >= bat_charge_min + bat_charge_rate
            if solar_pred(j) + bid_q - demand_pred(j) > 0
                bat_soc = min([bat_soc+bat_charge_rate, bat_soc+solar_pred(j)+bid_q-demand_pred(j), bat_cap]);
                grid_pred(j) = 0;
            elseif solar_pred(j) + bat_charge_rate + bid_q - demand_pred(j) > 0
                bat_soc = bat_soc - (demand_pred(j) - solar_pred(j) - bid_q);
                grid_pred(j) = 0;
            elseif solar_pred(j) + bat_charge_rate + bid_q - demand_pred(j) < 0
                bat_soc = bat_soc - bat_charge_rate;
                grid_pred(j) = demand_pred(j) - solar_pred(j) - bat_charge_rate - bid_q;
            end
        else
            if solar_pred(j) + bid_q - demand_pred(j) > 0
                bat_soc = min([bat_soc+bat_charge_rate, bat_soc+solar_pred(j)+bid_q-demand_pred(j), bat_cap]);
                grid_pred(j) = 0;
            else
                grid_pred(j) = demand_pred(j) - solar_pred(j) - bid_q;
            end
        end
    end
end