function new_weights = ann_train(state,weights,reward,q_sa,qnext_sa,gamma)
    
    eta1 = 0.01;
    eta2 = 0.001;
    eta3 = 0.0001;
    
    w1 = weights.w1;
    b1 = weights.b1;
    w2 = weights.w2;
    b2 = weights.b2;
    w3 = weights.w3;
    b3 = weights.b3;


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
    
	w1_deltas = eta1 * (state' * h_delta1);
	b1_deltas = eta1 * h_delta1;
	w2_deltas = eta2 * (h_out1' * h_delta2);
	b2_deltas = eta2 * h_delta2;
	w3_deltas = eta3 * (h_out2' * out_delta);
	b3_deltas = eta3 * out_delta;
    
	new_w1 = w1 + w1_deltas;
	new_b1 = b1 + sum(b1_deltas',2);
	new_w2 = w2 + w2_deltas;
	new_b2 = b2 + sum(b2_deltas',2);
	new_w3 = w3 + w3_deltas;
	new_b3 = b3 + sum(b3_deltas',2);
    
    new_weights = struct('w1',new_w1,'w2',new_w2,'w3',new_w3,'b1',new_b1,'b2',new_b2,'b3',new_b3);
    
end