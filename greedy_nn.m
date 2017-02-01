%% Function: greedy selection of action
function [q,value] = greedy_nn(state,weights,env_params)

    bid_q = env_params.min_bid_q:1:env_params.max_bid_q;
    q = randsample(bid_q,1);
    value = ann_pred([state,ones(size(state,1),1)*q/env_params.max_bid_q],weights);
    % Interating over all possible actions and finding the one 
    % with best q value
   for i=1:size(bid_q,2)
       newvalue = ann_pred([state,ones(size(state,1),1)*bid_q(i)/env_params.max_bid_q],weights);
       idx = find(newvalue>value);
       value(idx) = newvalue(idx);
       q(idx) = bid_q(i);
   end
end