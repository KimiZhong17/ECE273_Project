%% Blind Deconvolution via Non-convex Optimization
% Notation is based on the paper "Rapid, Robust, and Reliable Blind 
% Deconvolution via Nonconvex Optimization" by Xiaodong Li, Shuyang Ling, 
% Thomas Strohmer, and Ke Wei, from
% June 16, 2016
%
% Written by Guoren Zhong <gzhong@eng.ucsd.edu> 
% May 23, 2020
%% METHOD
% Condsider a blind deconvoltuion model: 
%   y = f * g (without noise), where y is given, but f and g are unknown.
% Now we assume that f in CC^L as (CC stands for complex space):
%   f = [h^T 0^T]^T, where h in CC^K, i.e. only the first K entries of f are
% nonzero and fl = 0 for all l = K+1, K+2, ..., L.
% Then, we assume that g belongs to a linear subspace spanned by the 
% columns of a known matrix C, i.e., 
%   g = C x_bar, where C is a Gaussian random LxN matrix.
% 
% Define h0 and x0 as the true (optimal) values (assuming ||h0|| = ||x0||). 
% Then, it would be convenient to transform this problem into Fourier domain.
% Let F be the Fourier transformation matrix, then B = F(:,1:K), so we have
%   sqrt(L) F y = diag(sqrt(L) F f)(sqrt(L) F g)  
%   with F f = B h ===> y = diag(B h) A x_bar, where A = F C.
%
% Goal: recover h0 and x0 when B, A, y is given.
%
% Define linear operator AA and its conjugate AA_star:
%   AA: AA(Z) = {b_l_star Z a_l}_{l=1}^L   
%   AA_star (z) = sum_{l=1}^L z_l b_l a_l_star
%
% Then we solve the problem by Wirtinger gradient descent
%   1. Compute AA_star(y)
%   2. Do svd to AA_star(y), and find the leading singular value d, and
%      leading left and right singular vector h0_hat and x0_hat.
%   3. Get the initialization values: 
%      u_0 = sqrt(d) * h0_hat, v_0 = sqrt(d) * conj(x0_hat);
%   4. Then, do gradient descent for t = 1,2,...,T
%      u_t = u_{t-1} - eta nabla Fh_tilde(u_{t-1}, v_{t-1})
%      v_t = v_{t-1} - eta nabla Fx_tilde(u_{t-1}, v_{t-1})
% The final recovery is given by u_T and v_T.
%% 
clear;
clc
global L B A K N y d mu rho;
rng(10);

%% Parameters 
N = 10; 
K = 10;
L = round(1.2 * (K + N));
T = 2000; % #iteration of gradient descent

%% signal of interest and blurring function
x = randn(N,1);
x = x/norm(x);
h = randn(K,1);
h = h/norm(h);

%% Fourier domain and transform matrices
F = dftmtx(L)/sqrt(L);
C = 1 * randn(L,N) + 1 * 1i * randn(L,N);
A = F * C;
B = F(:,1:K);
f = [h;zeros(L-K,1)];
g = C * x;

%% Convolve of the original unknown signals
y = OperatorA(h * x');

%% Initialization
B_star = B';
A_star = A';
Astar_y = OperatorA_star(y);
[Left,S,Right] = svd(Astar_y);
h0_hat = Left(:,1);
x0_hat = Right(:,1);
d = S(1,1);
mu = 6 * sqrt(L/(K+N)) / log(L);
rho = d^2/100;

u0 = sqrt(d) * h0_hat;
v0 = sqrt(d) * conj(x0_hat);

%% Projection
% We did not do this part because the simulation in the paper also skip
% this.
% z0 = randn(K,1);
% fun = @(z)norm(z-sqrt(d).*h0_hat)^2;
% [z,fval] = fmincon(fun,z0,[],[],[],[],[],[],@constraint)

% u0 = z;
% v0 = sqrt(d) * x0_hat;
%% Gradient descent
eta = 1/((N*log(L)+ rho*L/(mu^2)));
U = zeros(K,T);
V = zeros(N,T);
U(:,1) = u0/norm(u0);
V(:,1) = v0/norm(v0);

for t=2:T
    U(:,t) = U(:,t-1) - eta * (nablaF_h(U(:,t-1),V(:,t-1)) + nablaG_h(U(:,t-1)));
    V(:,t) = V(:,t-1) - eta * (nablaF_x(U(:,t-1),V(:,t-1)) + nablaG_x(V(:,t-1)));
    U(:,t) = U(:,t)/norm(U(:,t));
    V(:,t) = V(:,t)/norm(V(:,t));
end

u_rec = U(:,T);
v_rec = V(:,T);

X = u_rec * v_rec';
error = norm(X - h*x','fro')/norm(h*x','fro')

%% Functions used in this problem
% this function is for the projection part
% function[c,ceq] = constraint(z)
%     global L B mu d;
%     ceq = [];
%     c = sqrt(L) .* norm(B*z,'Inf') - 2*sqrt(d) * mu;
% end

function [result] = OperatorA(Z)
    global A B;
    result = diag(B*Z*A');
end

function [result] = OperatorA_star(z)
    global K N L A B;
    A_star = A';
    B_star = B';
    result = zeros(K,N);
    for i=1:L
        result = result + z(i) * B_star(:,i) * A_star(:,i)';
    end
end

function [gradient] = nablaF_h(h, x)
    global y;
    gradient = OperatorA_star(OperatorA(h * x') - y) * x;
end

function [gradient] = nablaF_x(h, x)
    global y;
    gradient = OperatorA_star(OperatorA(h * x') - y)' * h;
end

function [G0p] = G0_p(z)
    G0 = max(z-1,0);
    G0 = G0^2;
    G0p = 2*sqrt(G0);
end

function [gradient] = nablaG_h(h)
    global rho L d mu B;
    temp = 0;
    B_star = B';
    for i=1:L
        temp = temp + G0_p(L*abs(B_star(:,i)'*h)^2/(8*d*mu^2)) * B_star(:,i) * B_star(:,i)';
    end
    gradient = (rho/(2*d)) * (G0_p(norm(h)^2/(2*d))*h + (L/(4*mu^2))*temp*h);
end

function [gradient] = nablaG_x(x)
    global rho d;
    gradient = (rho/(2*d)) * G0_p(norm(x)^2/(2*d)) * x;
end