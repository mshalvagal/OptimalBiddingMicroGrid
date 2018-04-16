function bid_q_o = static_sim(i,bat_soc,mode,env_params,energy_data)
    
    NBlocks = env_params.NBlocks;
    grid_rate = env_params.grid_rate;
    bat_cap = env_params.bat_params.bat_cap;
    bat_charge_rate = env_params.bat_params.bat_charge_rate;
    bat_charge_min = env_params.bat_params.bat_charge_min;
    
    if strcmp(mode,'predicted')
        demand = energy_data.demand_pred(i,:);
        solar = energy_data.solar_pred(i,:);
        acp_for_day = [mean(energy_data.acp_a(i-30:i-1,:));1:NBlocks];
    elseif strcmp(mode,'actual')
        demand = energy_data.demand_a(i,:);
        solar = energy_data.solar_a(i,:);
        acp_for_day = [energy_data.acp_a(i,:);1:NBlocks];
    end
    
    bid_q_o = zeros(1,NBlocks);
%     bat_charge_agg = bat_soc*ones(NBlocks,1);
%     usedBlocks = 0;
    
    for i = 1:NBlocks
        if acp_for_day<0.7*grid_rate
            if demand(i)<solar(i)
                bid_q_o(i) = max(0,bat_charge_rate-(solar(i)-demand(i)));
            else
                bid_q_o(i) = demand(i)-solar(i)+bat_charge_rate;
            end
        else
            if demand(i)<solar(i)
                bid_q_o(i) = solar(i)-demand(i);
            else
                bid_q_o(i) = 0;
            end
        end
    end
    
%     while(~isempty(acp_for_day))
%         [~,maxIndex] = max(acp_for_day(1,:));
%         if maxIndex==1            
%             acp_for_day(:,maxIndex) = [];
%             continue;
%         end
%         maxBlock = acp_for_day(2,maxIndex);
%         [acp_temp,minIndex] = min(acp_for_day(1,1:maxIndex-1));
%         minBlock = acp_for_day(2,minIndex);
%         if acp_temp > grid_rate
%             break;
%         end
%         
%         bid_q_o(minBlock) = demand(minBlock)-solar(minBlock) + bat_charge_rate;
%         bid_q_o(maxBlock) = demand(maxBlock)-solar(maxBlock) - bat_charge_rate;
%         bid_q_o(bid_q_o>env_params.max_bid_q) = env_params.max_bid_q;
%         bid_q_o(bid_q_o<env_params.min_bid_q) = env_params.min_bid_q;
%         
%         bat_charge_agg(minBlock+1:maxBlock-1) = bat_charge_agg(minBlock+1:maxBlock-1) + min(bat_charge_rate,demand(minBlock)-solar(minBlock)-bid_q_o(minBlock));
%         
%         if max(bat_charge_agg)>bat_cap || min(bat_charge_agg)<bat_charge_min
% %             acp_for_day(:,minIndex:maxIndex) = [];
% %             usedBlocks = usedBlocks + maxIndex - minIndex + 1;
%             break;
%         else
%             acp_for_day(:,minIndex) = [];
%             acp_for_day(:,maxIndex-1) = [];
%             usedBlocks = usedBlocks + 2;
%         end
%     end
end