%% BLIND DECONVOLUTION
% Notation is based on the paper "Blind Deconvolution using Convex 
% Programming", by Ali Ahmed, Benjamin Recht, and Justin Romberg, from 
% July 22, 2013
%
% Written by Guoren Zhong <gzhong@eng.ucsd.edu> 
% May 15, 2020
%% METHOD: 
% We now have 2 signals w and x, and 
%   w = B h
%   x = C m
% By the circular convolution (denoted as * here), we have 
%   y = w * x  <==>  y(l) = sum_l' w(l') x(l-l'+1)
%   ==> y = m(1) w * C_1 + ... + m(N) w * C_N 
%         = [circ(C_1) ... circ(C_N)] [m(1) w ... m(N) w]'
%         = [circ(C_1) B ... circ(C_N) B] [m(1) h ... m(N) h]'
% Introduce the Fourier domain: F
%   B_hat = F B
%   C_hat = F C
%   circ(C_n) = F' D_n F,   D_n = diag(sqrt(L) C_hat_n)
%   ==> y_hat = F y = [D_1 B_hat ... D_N B_hat] [m(1) h ... m(N) h]'
% Then, the blind deconv problem === a linear inverse problem to recover 
% a K x N matrix from observation y_hat = A (X_0)
%   - X_0 is a rank 1 matrix
%   - A: K x N --> L
% For b_hat_l being the l-th column of B_hat', and c_hat_l being the l-th 
% row of sqrt(L) C_hat, then each entry of y_hat is translated to:
%   y_hat(l) = c_hat_l(1) m(1) <h, b_hat_l> + ... + c_hat_l(N) m(N) <h, b_hat_l>
%            = <c_hat_l, m> <h, b_hat_l>
%            = trace(A_l'(h m')),    A_l = b_hat_l c_hat_l'
%
% Given y_hat, our goal is to find h and m:
%   ______________________________________________________________
%  | min_{m,h} ||m||^2 + ||h||^2                                  |
%  |    s.t.   y_hat(l) = <c_hat_l, m> <h, b_hat_l>,  l = 1,...,L |
%   --------------------------------------------------------------- 
% By doing the "dual-dual" relaxation, we have:
%   ____________________
%  |  min ||X||_*       |
%  | s.t. y_hat = A (X) |
%   --------------------

%% 
clear all;
clc
rng(1);

%% Parameters 
N = 10; 
K = 10;
% L = round(3 * (K + N));
L = 2^9;

%% Randomly generate vectors m and h
m = randn(N,1);
m = m/norm(m);
h = randn(K,1);
h = h/norm(h);

%% Generating B and C, then calculate w and x
idxB = randperm(L);
idxB = idxB(1:K);
B = eye(L); % sparsity condition
% B = randn(L,L); % violating sparsity
B = B(:,idxB); % sparse
% B = B(:,1:K); % short
w = B * h;

idxC = randperm(L);
idxC = idxC(1:N);
C = eye(L);
C = C(:,idxC); % sparse
x = C * m;

%% Convolve x and w
y = real(ifft(fft(x).*fft(w)));

%% Convert to Fourier domain
B_hat = fft(B);
C_hat = fft(C);
y_hat = fft(y);

%% Define the linear operator A
A = zeros(L,K*N);
for i=1:size(C_hat,2)
    Delta = diag(sqrt(L)*C_hat(:,i));
    A(:,(i-1)*K+1:i*K) = Delta * B_hat;
end

%% CVX solver
cvx_begin
    variable X(K,N) 
    minimize( norm_nuc(X) )
    subject to
        A*X(:) == y_hat
cvx_end

%% Deconvolution
[U,S,V] = svd(X);
u = U(:,1);
v = V(:,1);
error = norm(u*v' - h*m','fro')/norm(h*m','fro')