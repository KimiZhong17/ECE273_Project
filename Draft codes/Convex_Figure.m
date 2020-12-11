P_success = zeros(25,25);
filename = sprintf('Phase_transitions');
L = 2^8;

for k=1:25
    k
    for j=1:25
        N = 5 * k;
        K = 5 * j;
        
        m = randn(N,1);
        m = m/norm(m);
        h = randn(K,1);
        h = h/norm(h);
        
        idxB = randperm(L);
        idxB = idxB(1:K);
        B = randn(L,L);
        B = B(:,idxB);
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
        P_success(k,j) = error;
    end
end
%%
P_success_dense_cut = P_success;
for i=1:25
    for j=1:25
        if P_success_dense_cut(i,j) < 10e-2
            P_success_dense_cut(i,j) = 1;
        elseif P_success_dense_cut(i,j) > 10
            P_success_dense_cut(i,j) = 0;
        else
            P_success_dense_cut(i,j) = 1/P_success_dense_cut(i,j)/10;
        end
    end
end
%P_success_dense_cut(22,1) = 1;
%Emax = max(max(P_success_short_cut));
%Emin = min(min(P_success_short_cut));
%%
imagesc(P_success_dense_cut);
colormap(gray);
set(gca,'YDir','normal');
set(gca,'xticklabel',{'25','50','75','100','125'});
set(gca,'yticklabel',{'25','50','75','100','125'});
title('L = 256');
xlabel('K');
ylabel('N');
set(gca,'FontName','Times New Roman','FontSize',12)
grid on;
grid minor;
colorbar('Fontsize',11);