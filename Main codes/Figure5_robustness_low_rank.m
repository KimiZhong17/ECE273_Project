%% Experiment 1 for violating low-rank condition
P_success_high_rank = zeros(16,1);
Error_high_rank = zeros(16,50);

N = 7;
K = 13;
for coe = 1:16
    coe
    for rd = 1:50
        rng(rd);
        L = round((0.2 * coe + 0.8) * (K + N));
        m = randn(N,1);
        m = m/norm(m);
        h = randn(K,1);
        h = h/norm(h);

        idxB = randperm(L);
        idxB = idxB(1:K);
        B = eye(L,L);
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
        Error_high_rank(coe,rd) = error;
        if error<0.02
            P_success_high_rank(coe) = P_success_high_rank(coe) + 0.02;
        end
    end
end
%% Plot Fig.5(a)
plot(linspace(1,16,16),P_success_high_rank,'-p',linspace(1,16,16),P_success_convex,'-o');
xlabel('L/(K+N)');
ylabel('Probability of successful recovery');
xlim([1,16]);
ylim([-0.05, 1.05]);
xticks(linspace(1,16,7));
set(gca,'xticklabel',{'1','1.5','2','2.5','3','3.5','4'});
legend('K = 13, N = 7','K = N = 10','Location','southeast');
%title('Rank');
set(gca,'FontName','Times New Roman','FontSize',12);
grid on;


%% Experiment 2 for violating low-rank condition
P_success_high_rank_ex2 = zeros(15,1);
Error_high_rank_ex2 = zeros(15,50);

N = 10;
for coe = 1:15
    coe
    for rd = 1:50
        rng(rd);
        K = round(coe + 5);
        L = 3*(K + N);
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
            minimize( norm_nuc(X))
            subject to
                A*X(:) == y_hat;
        cvx_end

        [U,S,V] = svd(X);
        u = U(:,1);
        v = V(:,1);
        error = norm(u*v' - h*m','fro')/norm(h*m','fro');
        Error_high_rank_ex2(coe,rd) = error;
        if error<0.02
            P_success_high_rank_ex2(coe) = P_success_high_rank_ex2(coe) + 0.02;
        end
    end
end
%% Plot Fig.5(b)
plot(linspace(1,15,15),P_success_high_rank_ex2,'-d');
xlabel('K');
ylabel('Probability of successful recovery');
xlim([1,15]);
ylim([-0.05, 1.05]);
xticks(linspace(0,15,6));
set(gca,'xticklabel',{'5','8','11','14','17','20'});
%legend('K','Location','southeast');
title('N = 10,  L = 3 (K + N)');
set(gca,'FontName','Times New Roman','FontSize',12);
grid on;