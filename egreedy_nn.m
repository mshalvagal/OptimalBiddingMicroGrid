%% Function: egreedy seletion of actions
function [q,value] = egreedy_nn(state,agent_params,env_params)

    bid_q = env_params.min_bid_q:1:env_params.max_bid_q;
    q = randsample(bid_q,1);
    value = ann_pred([state,q/env_params.max_bid_q],agent_params.weights);
    if rand()>=agent_params.epsilon
       [q,value] = greedy_nn(state,agent_params.weights,env_params);
    end
end