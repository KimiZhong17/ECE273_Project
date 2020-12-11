%%
P_success_convex = zeros(16,1);
P_success_nonconvex = zeros(16,1);
for coe = 1:16
    for rd = 1:50
        if Error_nonconvex(coe,rd)<0.02
            P_success_nonconvex(coe) = P_success_nonconvex(coe) + 0.02;
        end
    end
end
for coe = 1:16
    for rd = 1:50
        if Error_convex(coe,rd)<0.02
            P_success_convex(coe) = P_success_convex(coe) + 0.02;
        end
    end
end


%%
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