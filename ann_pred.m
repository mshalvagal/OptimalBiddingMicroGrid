function values = ann_pred(state,weights)
    
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
    values = h_out*w2 + B2;
end