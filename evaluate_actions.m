function [performance_measures,bat_storage] = evaluate_actions(i,bat_soc,bat_eff,env_params,energy_data,actions)

    NBlocks = env_params.NBlocks;
    grid_rate = env_params.grid_rate;
    bid_p = env_params.bid_p;
    
    bat_cap = env_params.bat_params.bat_cap;
    bat_charge_rate = env_params.bat_params.bat_charge_rate;
    bat_charge_min = env_params.bat_params.bat_charge_min;
    
    demand = energy_data.demand_a(i,:);
    solar = energy_data.solar_a(i,:);
    acp = energy_data.acp_a(i,:);
    
    grid=zeros(1,NBlocks);
    reward=zeros(1,NBlocks);
    reward2=zeros(1,NBlocks);
    wastage=zeros(1,NBlocks);
    bat_storage=zeros(1,NBlocks);
    bat_charge_cost=zeros(1,NBlocks);
    actual_cost=zeros(1,NBlocks);
    
    num_charge_cycles = 0;

    %% Actual energy usage for agent
    for j=1:NBlocks
        bat_storage(j) = bat_soc;
        bid_q = actions(j);
        % Check whether bid accepted or not and decide actual bid quantity
        if bid_p<acp(j)
            bid_q = 0;
        end
        %% Find actual battery soc
        if bat_soc >= bat_charge_min + bat_charge_rate
            if solar(j) + bid_q - demand(j) > 0
                new_bat_soc = min([bat_soc+bat_charge_rate, bat_soc+solar(j)+bid_q-demand(j), bat_cap]);
                if new_bat_soc==bat_cap
                    wastage(j) = bat_soc + solar(j) + bid_q - demand(j) - bat_cap;
                end
                grid(j) = 0;
            elseif solar(j) + bat_eff*bat_charge_rate + bid_q - demand(j) >= 0
                new_bat_soc = bat_soc - (demand(j) - solar(j) - bid_q)/bat_eff;
                grid(j) = 0;
            elseif solar(j) + bat_eff*bat_charge_rate + bid_q - demand(j) < 0
                new_bat_soc = bat_soc - bat_charge_rate;
                grid(j) = demand(j) - solar(j) - bat_eff*bat_charge_rate - bid_q;
            end
        else
            if solar(j) + bid_q - demand(j) > 0
                new_bat_soc = min([bat_soc+bat_charge_rate, bat_soc+solar(j)+bid_q-demand(j), bat_cap]);
                if new_bat_soc==bat_cap
                    wastage(j) = bat_soc + solar(j) + bid_q - demand(j) - bat_cap;
                end
                grid(j) = 0;
            else
                new_bat_soc = bat_soc;
                grid(j) = demand(j) - solar(j) - bid_q;
            end
        end
        
        actual_cost(j) = bid_q*acp(j) + grid(j)*grid_rate;
        
        %% Finding reward based on actuals          
        reward(j) = - wastage(j);
%         reward(j) = 0;
%         reward2(j) = -actual_cost(j);
        reward2(j) = -(bid_q*acp(j) + grid(j)*grid_rate + wastage(j)*acp(j));
        
        if new_bat_soc>bat_soc
            bat_charge_cost(j) = acp(j)*(bat_soc-new_bat_soc);
%             reward(j) = reward(j) + bat_charge_cost(j);
        else
            bat_charge_cost(j) = grid_rate*(bat_soc-new_bat_soc);
            reward(j) = reward(j) + (bat_soc-new_bat_soc);
        end
        
        if new_bat_soc~=bat_soc
            num_charge_cycles = num_charge_cycles + 1;
        end
        
        bat_soc = new_bat_soc;
    end
    performance_measures = struct('grid',grid,'reward',reward,'wastage',wastage,'bat_charge_cost',bat_charge_cost,'actual_cost',actual_cost,'charge_cycles',num_charge_cycles,'reward2',reward2);
end