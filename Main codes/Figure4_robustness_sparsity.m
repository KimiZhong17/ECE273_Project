%% Dense B only
P_success_low_sparsity = zeros(16,1);
Error_low_sparsity = zeros(16,50);
N = 10;
K = 10;
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
        B = randn(L,L);
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
        Error_low_sparsity(coe,rd) = error;
        if error<0.02
            P_success_low_sparsity(coe) = P_success_low_sparsity(coe) + 0.02;
        end
    end
end
%% Dense B and C
P_success_low_sparsity_both = zeros(16,1);
Error_low_sparsity_both = zeros(16,50);
N = 10;
K = 10;
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
        B = randn(L,L);
        B = B(:,1:K); %short
        w = B * h;

        idxC = randperm(L);
        idxC = idxC(1:N);
        C = randn(L);
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
        Error_low_sparsity_both(coe,rd) = error;
        if error<0.02
            P_success_low_sparsity_both(coe) = P_success_low_sparsity_both(coe) + 0.02;
        end
    end
end

%% Plot Fig.4
plot(linspace(1,16,16),P_success_low_sparsity_both,'-s',linspace(1,16,16),P_success_low_sparsity,'-^',linspace(1,16,16),P_success_convex,'-o');
xlabel('L/(K+N)');
ylabel('Probability of successful recovery');
xlim([1,16]);
ylim([-0.05, 1.05]);
xticks(linspace(1,16,7));
set(gca,'xticklabel',{'1','1.5','2','2.5','3','3.5','4'});
legend('Dense B and C','Dense B and sparse C','Sparse B and C','Location','southeast');
title('K = N = 10');
set(gca,'FontName','Times New Roman','FontSize',12);
grid on;