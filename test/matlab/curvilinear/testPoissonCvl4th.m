function errnorm = testPoissonCvl4th(N)
    u = @(x,y) sin(pi*x) .* cos(pi*y);
    u_x = @(x,y) pi * cos(pi*x) .* cos(pi*y);
    u_y = @(x,y) -pi * sin(pi*x) .* sin(pi*y);
    ddu = @(x,y) (-2*pi*pi) * sin(pi*x) .* cos(pi*y);
    
    ioRadius = [1 2];
    maxTheta = pi/2;
    region = [ioRadius(2) - ioRadius(1) maxTheta];
    [H,dx,ciij] = genMetricOfSector(ioRadius, maxTheta, N);
    
    rspan = linspace(-dx(1)*(3/2), region(1) + dx(1)*(3/2), N(1)+4);
    tspan = linspace(-dx(2)*(3/2), region(2) + dx(2)*(3/2), N(2)+4);
    [R,T] = meshgrid(rspan, tspan);
    R = R';
    T = T';
    X = (R + ioRadius(1)) .* cos(T);
    Y = (R + ioRadius(1)) .* sin(T);
    
    err = u(X,Y);
    rhs = zeros(N(1)+4,N(2)+4);
    rhs(3:end-2,3:end-2) = ddu(X(3:end-2,3:end-2),Y(3:end-2,3:end-2));
    rhs(3,3) = u(X(3,3), Y(3,3)); % a dirichlet value
    % lo face of dimension 1
    xx = ioRadius(1) * cos(tspan(3:end-2));
    yy = ioRadius(1) * sin(tspan(3:end-2));
    nx = -xx ./ hypot(xx,yy);
    ny = -yy ./ hypot(xx,yy);
    rhs(2,3:end-2) = u_x(xx, yy) .* nx + u_y(xx, yy) .* ny;
    % hi face of dimension 1
    xx = ioRadius(2) * cos(tspan(3:end-2));
    yy = ioRadius(2) * sin(tspan(3:end-2));
    nx = xx ./ hypot(xx,yy);
    ny = yy ./ hypot(xx,yy);
    rhs(end-1,3:end-2) = u_x(xx, yy) .* nx + u_y(xx,yy) .* ny;
    % lo face of dimension 2
    xx = (rspan(3:end-2) + ioRadius(1)) * cos(0);
    yy = (rspan(3:end-2) + ioRadius(1)) * sin(0);
    rhs(3:end-2,2) = -u_y(xx, yy)';
    % hi face of dimension 2
    xx = (rspan(3:end-2) + ioRadius(1)) * cos(maxTheta);
    yy = (rspan(3:end-2) + ioRadius(1)) * sin(maxTheta);
    rhs(3:end-2,end-1) = -u_x(xx, yy)';
    %
    rhs = rhs(:);
    L = genLaplacianCvl4th(H,ciij,dx,'N');
    sol = L\rhs;
    sol = reshape(sol, N(1)+4, N(2)+4);
    sol = sol(3:end-2,3:end-2);
    err = sol - err(3:end-2,3:end-2);
    errnorm = norm(err(:), inf);
end

%
%        32        64        96       128
%  2.40e-03  1.91e-04  4.14e-05  1.38e-05
%    3.6514    3.7710    3.8188
