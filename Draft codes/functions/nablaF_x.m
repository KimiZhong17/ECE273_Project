function [gradient] = nablaF_h(h, x)
    global y;
    gradient = OperatorA_star(OperatorA(h * x') - y) * x;
end

function [gradient] = nablaF_x(h, x)
    global y;
    gradient = OperatorA_star(OperatorA(h * x') - y) * h;
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