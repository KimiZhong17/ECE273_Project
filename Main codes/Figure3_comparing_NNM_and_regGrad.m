%%
clear all;
clc
global L B A K N y d mu rho;
%% Convex Approach --- NNM
P_success_convex = zeros(16,1);
Error_convex = zeros(16,50);
N = 10;
K = 10;
for coe = 1:16
    for rd = 1:50
        rng(rd);
        L = round((0.2 * coe + 0.8) * (K + N));
        m = randn(N,1);
        m = m/norm(m);
        h = randn(K,1);
        h = h/norm(h);

        idxB = randperm(L);
        idxB = idxB(1:K);
        B = eye(L);
        B = B(:,1:K); %short
        w = B * h;

        idxC = randperm(L);
        idxC = idxC(1:N);
        C = eye(L);
        C = C(:,idxC);
        x = C * m;

        y = real(ifft(fft(x).*fft(w)));
        B_hat = fft(B);
        C_hat = fft(C);
        y_hat = fft(y);

        A = zeros(L,K*N);
        for i=1:size(C_hat,2)
            Delta = diag(sqrt(L)*C_hat(:,i));
            A(:,(i-1)*K+1:i*K) = Delta * B_hat;
        end

        cvx_begin
            variable X(K,N) 
            minimize( norm_nuc(X) )
            subject to
                A*X(:) == y_hat;
        cvx_end

        [U,S,V] = svd(X);
        u = U(:,1);
        v = V(:,1);
        error = norm(u*v' - h*m','fro')/norm(h*m','fro');
        Error_convex(coe,rd) = error;
        if error<0.02
            P_success_convex(coe) = P_success_convex(coe) + 0.02;
        end
    end
end

%% Nonconvex Approach --- regGrad
P_success_nonconvex = zeros(16,1);
Error_nonconvex = zeros(16,50);

for coe = 1:16
    for rd = 1:50
        rng(rd);
        N = 10; 
        K = 10;
        L = round((0.2 * coe + 0.8) * (K + N));
        T = 2000; % #iteration of gradient descent

        x = randn(N,1);
        x = x/norm(x);
        h = randn(K,1);
        h = h/norm(h);

        F = dftmtx(L)/sqrt(L);
        C = 1 * randn(L,N) + 1 * 1i * randn(L,N);
        A = F * C;
        %A = sqrt(0.5) * randn(L,N) + sqrt(0.5) * 1i * randn(L,N);
        %C = inv(F) * A;
        B = F(:,1:K);
        f = [h;zeros(L-K,1)];
        g = C * x;

        y = OperatorA(h * x');

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
        error = norm(X - h*x','fro')/norm(h*x','fro');
        Error_nonconvex(coe,rd) = error;
        if error<0.02
            P_success_nonconvex(coe) = P_success_nonconvex(coe) + 0.02;
        end
    end
end

%% Plot Fig.3
plot(linspace(1,16,16),P_success_nonconvex,'-*',linspace(1,16,16),P_success_convex,'-o');
xlabel('L/(K+N)');
ylabel('Probability of successful recovery');
xlim([1,16]);
ylim([-0.05, 1.05]);
xticks(linspace(1,16,7));
set(gca,'xticklabel',{'1','1.5','2','2.5','3','3.5','4'});
legend('regGrad','NNM','Location','southeast');
title('K = N = 10');
set(gca,'FontName','Times New Roman','FontSize',12);
grid on;

%% Functions used in this problem
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