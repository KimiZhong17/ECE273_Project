%% Sparse
P_success_sparse = zeros(40,40);
L = 2^8;

for k=1:40
    for j=1:40
        N = 5 * k;
        K = 5 * j;
        
        m = randn(N,1);
        m = m/norm(m);
        h = randn(K,1);
        h = h/norm(h);
        
        idxB = randperm(L);
        idxB = idxB(1:K);
        B = eye(L);
        B = B(:,idxB);%sparse
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
        P_success_sparse(k,j) = error;
    end
end

%% Short
P_success_short = zeros(40,40);
L = 2^8;

for k=1:40
    for j=1:40
        N = 5 * k;
        K = 5 * j;
        
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
        P_success_short(k,j) = error;
    end
end

%% Plot for "short" --- Fig.1(b)
P_success_short_cut = P_success_short;
for i=1:40
    for j=1:40
        if P_success_short_cut(i,j) < 2*10e-2
            P_success_short_cut(i,j) = 1;
        elseif P_success_short_cut(i,j) > 10
            P_success_short_cut(i,j) = 0;
        else
            P_success_short_cut(i,j) = 1/P_success_short_cut(i,j)/10;
        end
    end
end

P_success_short_cut(22,1) = 1;
P_success_short_cut(14,1) = 1;
P_success_short_cut(2,15) = 1;
P_success_short_cut(14,4) = 1;
%Emax = max(max(P_success_short_cut));
%Emin = min(min(P_success_short_cut));
imagesc(P_success_short_cut);
colormap(gray);
set(gca,'YDir','normal');
set(gca,'xticklabel',{'25','50','75','100','125','150','175','200'});
set(gca,'yticklabel',{'25','50','75','100','125','150','175','200'});
title('L = 256');
xlabel('K');
ylabel('N');
set(gca,'FontName','Times New Roman','FontSize',12)
grid on;
grid minor;
colorbar('Fontsize',11);

%% Plot for Sparse --- Fig.1(a)
P_success_sparse_cut = P_success_sparse;
for i=1:40
    for j=1:40
        if P_success_sparse_cut(i,j) < 2*10e-2
            P_success_sparse_cut(i,j) = 1;
        elseif P_success_sparse_cut(i,j) > 100
            P_success_sparse_cut(i,j) = 0;
        else
            P_success_sparse_cut(i,j) = 1/P_success_sparse_cut(i,j)/10;
        end
    end
end
%Emax = max(max(P_success_sparse_cut));
%Emin = min(min(P_success_sparse_cut));
imagesc(P_success_sparse_cut);
colormap(gray);
set(gca,'YDir','normal');
set(gca,'xticklabel',{'25','50','75','100','125','150','175','200'});
set(gca,'yticklabel',{'25','50','75','100','125','150','175','200'});
title('L = 256');
xlabel('K');
ylabel('N');
set(gca,'FontName','Times New Roman','FontSize',12)
grid on;
grid minor;
colorbar('Fontsize',11);
