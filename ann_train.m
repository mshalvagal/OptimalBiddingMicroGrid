function new_weights = ann_train(state,weights,reward,q_sa,qnext_sa,gamma)
    
    eta1 = 0.0001;
    eta2 = 0.00001;
    
    w1 = weights.w1;
    b1 = weights.b1;
    w2 = weights.w2;
    b2 = weights.b2;

    n_hid = size(w1,2);
    N = size(state,1);
    B1 = ones(N,n_hid)*diag(b1);
    B2 = ones(N,1)*diag(b2);
    
    h_in = state*w1 + B1;
	h_out = 1./(1 + exp(-h_in));
%     h_out = h_in;
%     h_out(h_in<0) = 0;
    
	out_derivatives = 1;
	out_error = reward + gamma*qnext_sa - q_sa;
	out_delta = out_derivatives .* out_error;
    
	h_derivatives = (exp(-h_in))./((1+exp(-h_in)).^2);
%     h_derivatives = ones(size(h_in));
%     h_derivatives(h_in<0) = 0;
	h_delta = (out_delta*w2');
	h_delta = h_derivatives .* h_delta;
    
	w1_deltas = eta1 * (state' * h_delta);
	b1_deltas = eta1 * h_delta;
	w2_deltas = eta2 * (h_out' * out_delta);
	b2_deltas = eta2 * out_delta;
    
	new_w1 = w1 + w1_deltas;
	new_b1 = b1 + sum(b1_deltas',2);
	new_w2 = w2 + w2_deltas;
	new_b2 = b2 + sum(b2_deltas',2);
    
    new_weights = struct('w1',new_w1,'w2',new_w2,'b1',new_b1,'b2',new_b2);
    
end