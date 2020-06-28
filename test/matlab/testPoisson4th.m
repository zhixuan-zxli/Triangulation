function errnorm = testPoisson4th(N, phi, dphi, f)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    h = 1.0/N;
    x = linspace(h/2, 1-h/2, N);
    x = x';
    %rhs = [0; -dphi(0); f(x); dphi(1); 0];
    rhs = [0; phi(0); f(x); phi(1); 0];
    L = 1.0/12/(h*h) * (diag(-30 * ones(N+4, 1), 0) ...
        + diag(16*ones(N+3,1), 1) ...
        + diag(16*ones(N+3,1), -1) ...
        + diag(-1*ones(N+2,1), 2) ...
        + diag(-1*ones(N+2,1), -2));
    L(1,:) = 0;
    L(1, 1:6) = [1 -5 10 -10 5 -1];
    L(2,:) = 0;
    %L(2,1:4) = 1.0/(24*h) * [-1 27 -27 1];
    L(2,2:6) = (35/128) * [1 4 -2 4/5 -1/7];
    L(N+3,:) = 0;
    %L(N+3,N+1:N+4) = 1.0/(24*h) * [1 -27 27 -1];
    L(N+3, N-1:N+3) = (35/128) * [-1/7 4/5 -2 4 1];
    L(N+4,:) = 0;
    L(N+4,N-1:N+4) = [1 -5 10 -10 5 -1];
%     L(3,:) = 0;
%     L(3,3) = 1;
%     rhs(3) = 0;
    
    sol_2 = L\rhs;
    sol = sol_2(3:end-2);
    err = phi(x);
    err = sol - err; % - (sol(3) - err(3));
    errnorm = norm(err,inf);
    
    % Pade differentiation
%     A = diag(ones(N+3,1),0) ...
%         + diag(1.0/22*ones(N+2,1), 1) ...
%         + diag(1.0/22*ones(N+2,1), -1);
%     A(1,:) = 0;
%     A(1,2) = 1;
%     A(N+3,:) = 0;
%     A(N+3,N+2) = 1;
%     b = (12/(11*h)) * (sol_2(3:N+3) - sol_2(2:N+2));
%     b = [dphi(0); b; dphi(1)];
%     dsol = A\b;

    % calculate the gradient using explicit differentiation
    dsol = 1.0/(24*h) * (sol_2(1:end-3) - 27 * sol_2(2:end-2) + 27 * sol_2(3:end-1) - sol_2(4:end));
    err_d = dsol - dphi(linspace(0,1,N+1)');
    errnorm = [errnorm; norm(err_d,inf)];
    % fill the ghost faces
    dsol = [0; dsol; 0];
    dsol(1) = 5 * dsol(2) - 10 * dsol(3) + 10 * dsol(4) - 5 * dsol(5) + dsol(6);
    dsol(end) = 5 * dsol(end-1) - 10 * dsol(end-2) + 10 * dsol(end-3) - 5 * dsol(end-4) + dsol(end-5);
    % calculate the laplacian using explicit differentiation
    ddsol = 1.0/(24*h) * (dsol(1:end-3) - 27 * dsol(2:end-2) + 27 * dsol(3:end-1) - dsol(4:end));
    err_dd = ddsol - f(x);
    errnorm = [errnorm; norm(err_dd(2:end-1), inf)];
end

