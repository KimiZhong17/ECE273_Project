function [result] = OperatorA(Z)
    global A B;
    result = diag(B*Z*A');
end