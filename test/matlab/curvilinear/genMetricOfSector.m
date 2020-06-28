function [H,dx,ciij] = genMetricOfSector(ioRadius, maxTheta, N)
% ioRadius = [innerRadius outerRadius]
% N = [Nx Ny] = subdivision number
%
% H{1}, H{2} : metric on the staggered faces.
% H{3} : metric at the cell centers.
% ciij : Christoffel symbol {i,i,j} of the second kind at the cell centers.
    region = [ioRadius(2) - ioRadius(1) maxTheta];
    dx = region ./ N;
    H = cell(1,3);
    
    % cell centers
    x = linspace(dx(1)/2, region(1) - dx(1)/2, N(1));
    y = linspace(dx(2)/2, region(2) - dx(2)/2, N(2));
    [X,Y] = meshgrid(x,y);
    X = X';
    Y = Y';
    H{3} = ones(N(1),N(2));
    H{3} = cat(3, H{3}, X+ioRadius(1));
    
    if nargout == 3
        ciij = 1.0 ./ (X+1);
        ciij = cat(3, ciij, zeros(N(1),N(2)));
    end
    
    % dimension 1
    x = linspace(0, region(1), N(1)+1);
    y = linspace(dx(2)/2, region(2) - dx(2)/2, N(2));
    [X,Y] = meshgrid(x,y);
    X = X';
    Y = Y';
    H{1} = ones(N(1)+1,N(2));
    H{1} = cat(3, H{1}, X+ioRadius(1));
    
    % dimension 2
    x = linspace(dx(1)/2, region(1) - dx(1)/2, N(1));
    y = linspace(0, region(2), N(2)+1);
    [X,Y] = meshgrid(x,y);
    X = X';
    Y = Y';
    H{2} = ones(N(1),N(2)+1);
    H{2} = cat(3, H{2}, X++ioRadius(1));
    
end

