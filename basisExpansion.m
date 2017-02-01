function [ state ] = basisExpansion( a, b, c, d, e, f )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    val = 10;
%     state = [a b c d e f d*a d*b d*c d^2 a*d^2 b*d^2 c*d^2 d^3 a*max(0, val - d) b*max(0, val - d) c*max(0, val - d) d*max(0, val - d) a*max(0, d - val) b*max(0, d - val) c*max(0, d - val) d*max(0, d - val) a*sqrt(d) b*sqrt(d) c*sqrt(d) d*sqrt(d) a*log(d) b*log(d) c*log(d) d*log(d)];
    state = [a b c d e f d*a d*b d*c d^2 a*d^2 b*d^2 c*d^2 d^3 a*log(d) b*log(d) c*log(d) d*log(d) a*sqrt(d) b*sqrt(d) c*sqrt(d) d*sqrt(d)];
end

