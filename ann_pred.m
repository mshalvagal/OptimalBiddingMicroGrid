function values = ann_pred(state,weights)
    
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
    h_in = state*w1 + B1;
    h_out = 1./(1 + exp(-h_in));
    h_in = h_out*w2 + B2;
    h_out = 1./(1 + exp(-h_in));
    values = h_out*w3 + B3;
end