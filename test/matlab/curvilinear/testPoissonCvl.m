function errnorm = testPoissonCvl(N, bctype)
    u = @(x,y) sin(pi*x) .* cos(pi*y);
    u_x = @(x,y) pi * cos(pi*x) .* cos(pi*y);
    u_y = @(x,y) -pi * sin(pi*x) .* sin(pi*y);
    ddu = @(x,y) (-2*pi*pi) * sin(pi*x) .* cos(pi*y);
    
    ioRadius = [1 2];
    maxTheta = pi/2;
    region = [ioRadius(2) - ioRadius(1) maxTheta];
    [H,dx] = genMetricOfSector(ioRadius, maxTheta, N);
    
    rspan = linspace(-dx(1)/2, region(1) + dx(1)/2, N(1)+2);
    tspan = linspace(-dx(2)/2, region(2) + dx(2)/2, N(2)+2);
    [R,T] = meshgrid(rspan, tspan);
    R = R';
    T = T';
    X = (R + ioRadius(1)) .* cos(T);
    Y = (R + ioRadius(1)) .* sin(T);
    
    err = u(X,Y);
    if strcmp(bctype, 'D')
        rhs = err;
        rhs(2:end-1,2:end-1) = ddu(X(2:end-1,2:end-1),Y(2:end-1,2:end-1));
        rhs = rhs(:);
        L = genLaplacianCvl(H, dx, 'D');
        sol = L\rhs;
        sol = reshape(sol, N(1)+2, N(2)+2);
        err = sol - err;
        errnorm = norm(err(:), inf);
    elseif strcmp(bctype, 'N')
        rhs = zeros(N(1)+2, N(2)+2);
        rhs(2:end-1,2:end-1) = ddu(X(2:end-1,2:end-1),Y(2:end-1,2:end-1));
        rhs(2,2) = u(X(2,2), Y(2,2)); % a dirichlet value
        % lo face of dimension 1
        xx = ioRadius(1) * cos(tspan);
        yy = ioRadius(1) * sin(tspan);
        nx = -xx ./ hypot(xx,yy);
        ny = -yy ./ hypot(xx,yy);
        rhs(1,:) = u_x(xx, yy) .* nx + u_y(xx, yy) .* ny;
        % hi face of dimension 1
        xx = ioRadius(2) * cos(tspan);
        yy = ioRadius(2) * sin(tspan);
        nx = xx ./ hypot(xx,yy);
        ny = yy ./ hypot(xx,yy);
        rhs(end,:) = u_x(xx, yy) .* nx + u_y(xx,yy) .* ny;
        % lo face of dimension 2
        xx = (rspan + ioRadius(1)) * cos(0);
        yy = (rspan + ioRadius(1)) * sin(0);
        rhs(:,1) = -u_y(xx, yy)';
        % hi face of dimension 2
        xx = (rspan + ioRadius(1)) * cos(maxTheta);
        yy = (rspan + ioRadius(1)) * sin(maxTheta);
        rhs(:,end) = -u_x(xx, yy)';
        rhs = rhs(:);
        L = genLaplacianCvl(H, dx, 'N');
        sol = L\rhs;
        sol = reshape(sol, N(1)+2, N(2)+2);
        sol = sol(2:end-1,2:end-1);
        err = sol - err(2:end-1,2:end-1);
        errnorm = norm(err(:), inf);
    end
end

%  Dirichlet
%          8        16        32        64
%     0.0566    0.0159    0.0040    0.0010
%     1.8318    1.9910    2.0058
%    Neumann
%          8        16        32        64       128
%     0.2830    0.0772    0.0208    0.0056    0.0015
%     1.8741    1.8920    1.8931    1.9121
