%% 
clear all;
clc
global L B A K N y d mu rho Astar_y;
rng(1);

%% Parameters 
N = 10; 
K = 10;
L = 2^9;

%% signal of interest and blurring function
x = randn(N,1);
x = x/norm(x);
h = randn(K,1);
h = h/norm(h);

%% Fourier domain
F = dftmtx(L)/sqrt(L);
A = 0.5 * randn(L,N) + 0.5 * 1i * randn(L,N);
C = inv(F) * A;
B = F(:,1:K);

%% Convolve of the original unknowns
f = [h;zeros(L-K,1)];
g = C * x;
y =diag(B*h)*A*x;


%%
B_star = B';
A_star = A';
Astar_y = h*x';
[Left,S,Right] = svd(Astar_y);
h0_hat = Left(:,1);
x0_hat = Right(:,1);
d = S(1,1);
mu = 6 * sqrt(L/(K+N)) / log(L);
rho = d^2/100;

%% 
z0 = randn(K,1);
fun = @(z)norm(z-sqrt(d).*h0_hat)^2;
[z,fval] = fmincon(fun,z0,[],[],[],[],[],[],@constraint)

u0 = z;
v0 = sqrt(d) * x0_hat;
%%
T = 50;
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
app_error = norm(real(OperatorA_star(y)) - h*x','fro')/norm(h*x','fro')

%%

function[c,ceq] = constraint(z)
    global L B mu d;
    ceq = [];
    c = sqrt(L) .* norm(B*z,'Inf') - 2*sqrt(d) * mu;
end

function [gradient] = nablaF_h(h, x)
    global Astar_y;
    gradient = (h * x' - Astar_y) * x;
end

function [gradient] = nablaF_x(h, x)
    global Astar_y;
    gradient = (h * x' - Astar_y) * h;
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