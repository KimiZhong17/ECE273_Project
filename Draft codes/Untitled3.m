global L B A K N y;
L = 2^9;
K = 10;
N = 10;
h = randn(1,K);
h = h/norm(h);
f = [h zeros(1,L-K)];
f = f';
C = randn(L,N);
x = randn(N,1);
x = x/norm(x);
g = C*x;
F = dftmtx(L)/sqrt(L);
B = F(:,1:K);
A = 0.5*randn(L,N)+0.5i*randn(L,N);
y = real(ifft(fft(f).*fft(g)));


[left,sigma,right] = svd(OperatorA_star(y,B',A'));
d = sigma(1);
global miu rho;
miu = 6*sqrt(L/(K+N))/log(L);
rho = d^2/100;
h0_hat = left(:,1);
x0_hat = right(:,1);
z0 = randn(length(h0_hat),1);
fun = @(z)norm(z-sqrt(d).*h0_hat)^2;
nonlcon = @(z)sqrt(L).*norm(B*z,'Inf');

opts = optimset('Display','iter','Algorithm','interior-point', 'MaxIter', 10000, 'MaxFunEvals', 10000);
[z,fval] = fmincon(fun,z0,[],[],[],[],[],[],@confuneq)
v0 = sqrt(d)*x0_hat;
u0 = fval;