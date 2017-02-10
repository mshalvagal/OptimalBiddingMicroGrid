function new_agent_params = ann_train(state,agent_params,reward,q_sa,qnext_sa,gamma)
    
    eta1 = 0.001;
    eta2 = 0.001;
    eta3 = 0.001;
    
    sq_grads = agent_params.sq_grads;
    w1 = agent_params.weights.w1;
    b1 = agent_params.weights.b1;
    w2 = agent_params.weights.w2;
    b2 = agent_params.weights.b2;
    w3 = agent_params.weights.w3;
    b3 = agent_params.weights.b3;


    n_hid1 = size(w1,2);
    n_hid2 = size(w2,2);
    N = size(state,1);
    B1 = ones(N,n_hid1)*diag(b1);
    B2 = ones(N,n_hid2)*diag(b2);
    B3 = ones(N,1)*diag(b3);
    
    
    h_in1 = state*w1 + B1;
    h_out1 = 1./(1 + exp(-h_in1));
    h_in2 = h_out1*w2 + B2;
    h_out2 = 1./(1 + exp(-h_in2));
%     h_out = h_in;
%     h_out(h_in<0) = 0;
    
	out_derivatives = 1;
	out_error = reward + gamma*qnext_sa - q_sa;
	out_delta = out_derivatives .* out_error;
    
	h_derivatives1 = (exp(-h_in1))./((1+exp(-h_in1)).^2);
	h_derivatives2 = (exp(-h_in2))./((1+exp(-h_in2)).^2);
%     h_derivatives = ones(size(h_in));
%     h_derivatives(h_in<0) = 0;
	h_delta2 = (out_delta*w3');
	h_delta2 = h_derivatives2 .* h_delta2;
	h_delta1 = (h_delta2*w2');
	h_delta1 = h_derivatives1 .* h_delta1;
    
	w1_deltas = eta1 * (state' * h_delta1)./sqrt(sq_grads.v_w1+1e-8);
	b1_deltas = eta1 * sum(h_delta1',2)./sqrt(sq_grads.v_b1+1e-8);
	w2_deltas = eta2 * (h_out1' * h_delta2)./sqrt(sq_grads.v_w2+1e-8);
	b2_deltas = eta2 * sum(h_delta2',2)./sqrt(sq_grads.v_b2+1e-8);
	w3_deltas = eta3 * (h_out2' * out_delta)./sqrt(sq_grads.v_w3+1e-8);
	b3_deltas = eta3 * sum(out_delta',2)./sqrt(sq_grads.v_b3+1e-8);
    
	new_w1 = w1 + w1_deltas;
	new_b1 = b1 + b1_deltas;
	new_w2 = w2 + w2_deltas;
	new_b2 = b2 + b2_deltas;
	new_w3 = w3 + w3_deltas;
	new_b3 = b3 + b3_deltas;
    
    sq_grads.v_w1 = 0.9*sq_grads.v_w1 + 0.1*w1_deltas.^2;
    sq_grads.v_w2 = 0.9*sq_grads.v_w2 + 0.1*w2_deltas.^2;
    sq_grads.v_w3 = 0.9*sq_grads.v_w3 + 0.1*w3_deltas.^2;
    sq_grads.v_b1 = 0.9*sq_grads.v_b1 + 0.1*b1_deltas.^2;
    sq_grads.v_b2 = 0.9*sq_grads.v_b2 + 0.1*b2_deltas.^2;
    sq_grads.v_b3 = 0.9*sq_grads.v_b3 + 0.1*b3_deltas.^2;
    
    new_weights = struct('w1',new_w1,'w2',new_w2,'w3',new_w3,'b1',new_b1,'b2',new_b2,'b3',new_b3);
    new_agent_params = agent_params;
    new_agent_params.weights = new_weights;
    new_agent_params.sq_grads = sq_grads;
end