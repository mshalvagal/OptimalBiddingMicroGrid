%% Function: greedy
function [q,value] = greedy(demand,solar,bat_soc,min_bid_q,max_bid_q,bid_p,block,tetha)
    bid_q = min_bid_q:1:max_bid_q;
    q = bid_q(ceil(rand()*size(bid_q,2)));
    state = basisExpansion(demand,solar,bat_soc,q/max_bid_q,bid_p,block);
    value = valuefn(state,tetha);
    % Interating over all possible actions and finding the one 
    % with best q value
   for i=1:size(bid_q,2)
       state = basisExpansion(demand,solar,bat_soc,bid_q(i)/max_bid_q,bid_p,block);
       newvalue = valuefn(state,tetha);
       if newvalue>value
           value=newvalue;
           q = bid_q(i);
       end
   end
   return;
end