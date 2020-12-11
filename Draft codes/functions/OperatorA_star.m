function [result] = OperatorA_star(z)
    global K N L A B;
    A_star = A';
    B_star = B';
    result = zeros(K,N);
    for i=1:L
        result = result + z(i) * B_star(:,i) * A_star(:,i)';
    end
end