%%
clc;
clear;
%% Nonblind Approach
P_success_nonblind = zeros(16,1);
Error_nonblind = zeros(16,50);
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
        %B = B(:,idxB); % sparse
        B = B(:,1:K); % short
        w = B * h;

        idxC = randperm(L);
        idxC = idxC(1:N);
        C = eye(L);
        C = C(:,idxC); % sparse
        x = C * m;

        y = real(ifft(fft(x).*fft(w)));
        B_hat = fft(B);
        C_hat = fft(C);
        y_hat = fft(y);

        z0 = [randn(K,1);zeros(L-K,1)];
        fun = @(z) norm(real(ifft(fft(x).*fft(z))) - y);
        [z,fval] = fmincon(fun,z0);
        error = norm(z*x' - w*x','fro')/norm(w*x','fro');
        if error<0.02
            P_success_nonblind(coe) = P_success_nonblind(coe) + 0.02;
        end
    end
end
%% Plot Fig.2
% To plot this, please import the data for NNM
plot(linspace(1,16,16),P_success_nonblind,'-x',linspace(1,16,16),P_success_convex,'-o');
xlabel('L/(K+N)');
ylabel('Probability of successful recovery');
xlim([1,16]);
ylim([-0.05, 1.05]);
xticks(linspace(1,16,7));
set(gca,'xticklabel',{'1','1.5','2','2.5','3','3.5','4'});
legend('Non-blind','NNM','Location','southeast');
title('K = N = 10');
set(gca,'FontName','Times New Roman','FontSize',12);
grid on;