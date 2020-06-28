function L = genLaplacianCvl4th(H, ciij, dx, bctype)
%genLaplacianCvl4th : Generate the 4th-order Laplacian matrix 
% on the curvilinear mesh whose metric is defined by H. 
% H : see genMetricOfSector.
% ciij : see genMetricOfSector.
    N = size(H{3});
    numUnk = (N(1)+4) * (N(2)+4);
    toIdx = @(i,j) j .* (N(1)+4) + i + N(1) + 6;
    L = zeros(numUnk);
    % interior nodes
    for j=1:N(2)
        for i=1:N(1)
            k = toIdx(i, j);            
            % dimension 1
            h1 = H{3}(i,j,1);
            a = ciij(i,j,1);
            L(k,k-2) = 1.0/(h1.^2) * (-1/(12*dx(1).^2) + a * 1/(12*dx(1)));
            L(k,k-1) = 1.0/(h1.^2) * (16/(12*dx(1).^2) + a * (-8)/(12*dx(1)));
            L(k,k) = 1.0/(h1.^2) * (-30/(12*dx(1).^2));
            L(k,k+1) = 1.0/(h1.^2) * (16/(12*dx(1).^2) + a * 8/(12*dx(1)));
            L(k,k+2) = 1.0/(h1.^2) * (-1/(12*dx(1).^2) + a * (-1)/(12*dx(1)));
            % dimension 2
            h2 = H{3}(i,j,2);
            a = ciij(i,j,2);
            L(k,k-2*(N(1)+4)) = 1.0/(h2.^2) * (-1/(12*dx(2).^2) + a * 1/(12*dx(2)));
            L(k,k-(N(1)+4)) = 1.0/(h2.^2) * (16/(12*dx(2).^2) + a * (-8)/(12*dx(2)));
            L(k,k) = L(k,k) + 1.0/(h2.^2) * (-30/(12*dx(2).^2));
            L(k,k+(N(1)+4)) = 1.0/(h2.^2) * (16/(12*dx(2).^2) + a * 8/(12*dx(2)));
            L(k,k+2*(N(1)+4)) = 1.0/(h2.^2) * (-1/(12*dx(2).^2) + a * (-1)/(12*dx(2)));
        end
    end
    
    % extrapolation conditions
    for j=1:N(2) % low face of dimension 1
        k = toIdx(-1,j);
        L(k,k) = 1; L(k,k+1) = -5; L(k,k+2) = 10;
        L(k,k+3) = -10; L(k,k+4) = 5; L(k,k+5) = -1;
    end
    for j=1:N(2) % high face of dimension 1
        k = toIdx(N(1)+2,j);
        L(k,k) = 1; L(k,k-1) = -5; L(k,k-2) = 10;
        L(k,k-3) = -10; L(k,k-4) = 5; L(k,k-5) = -1;
    end
    for i=1:N(1) % low face of dimension 2
        k = toIdx(i,-1);
        d = N(1)+4;
        L(k,k) = 1; L(k,k+d) = -5; L(k,k+2*d) = 10;
        L(k,k+3*d) = -10; L(k,k+4*d) = 5; L(k,k+5*d) = -1;
    end
    for i=1:N(1) % high face of dimension 2
        k = toIdx(i,N(2)+2);
        d = N(1)+4;
        L(k,k) = 1; L(k,k-d) = -5; L(k,k-2*d) = 10;
        L(k,k-3*d) = -10; L(k,k-4*d) = 5; L(k,k-5*d) = -1;
    end
    
    % Neumann boundary conditions
    if ~strcmp(bctype, 'N')
        error('Accept Neumann boundary condition only.');
    end
    for j=1:N(2) % low face of dimension 1
        k = toIdx(0,j);
        L(k,k-1) = -1 / (H{1}(1,j,1) * 24*dx(1));
        L(k,k)   = 27 / (H{1}(1,j,1) * 24*dx(1));
        L(k,k+1) = -27 / (H{1}(1,j,1) * 24*dx(1));
        L(k,k+2) = 1 / (H{1}(1,j,1) * 24*dx(1));
    end
    for j=1:N(2) % high face of dimension 1
        k = toIdx(N(1)+1,j);
        L(k,k+1) = -1 / (H{1}(N(1)+1,j,1) * 24*dx(1));
        L(k,k)   = 27 / (H{1}(N(1)+1,j,1) * 24*dx(1));
        L(k,k-1) = -27 / (H{1}(N(1)+1,j,1) * 24*dx(1));
        L(k,k-2) = 1 / (H{1}(N(1)+1,j,1) * 24*dx(1));
    end
    for i=1:N(1) % low face of dimension 2
        k = toIdx(i,0);
        d = N(1)+4;
        L(k,k-d)   = -1 / (H{2}(i,1,2) * 24*dx(2));
        L(k,k)     = 27 / (H{2}(i,1,2) * 24*dx(2));
        L(k,k+d)   = -27 / (H{2}(i,1,2) * 24*dx(2));
        L(k,k+2*d) = 1 / (H{2}(i,1,2) * 24*dx(2));
    end
    for i=1:N(1) % high face of dimension 2
        k = toIdx(i,N(2)+1);
        d = N(1)+4;
        L(k,k+d)   = -1 / (H{2}(i,N(2)+1,2) * 24*dx(2));
        L(k,k)     = 27 / (H{2}(i,N(2)+1,2) * 24*dx(2));
        L(k,k-d)   = -27 / (H{2}(i,N(2)+1,2) * 24*dx(2));
        L(k,k-2*d) = 1 / (H{2}(i,N(2)+1,2) * 24*dx(2));
    end
    
    % reset the four corners
    k = toIdx(-1,-1); L(k,k) = 1.0;
    k = toIdx(0,-1);  L(k,k) = 1.0;
    k = toIdx(-1,0);  L(k,k) = 1.0;
    k = toIdx(0,0);   L(k,k) = 1.0;
    k = toIdx(N(1)+1,-1); L(k,k) = 1.0;
    k = toIdx(N(1)+2,-1); L(k,k) = 1.0;
    k = toIdx(N(1)+1,0);  L(k,k) = 1.0;
    k = toIdx(N(1)+2,0);  L(k,k) = 1.0;
    k = toIdx(-1,N(2)+1); L(k,k) = 1.0;
    k = toIdx(0,N(2)+1);  L(k,k) = 1.0;
    k = toIdx(-1,N(2)+2); L(k,k) = 1.0;
    k = toIdx(0,N(2)+2);  L(k,k) = 1.0;
    k = toIdx(N(1)+1,N(2)+1); L(k,k) = 1.0;
    k = toIdx(N(1)+2,N(2)+1); L(k,k) = 1.0;
    k = toIdx(N(1)+1,N(2)+2); L(k,k) = 1.0;
    k = toIdx(N(1)+2,N(2)+2); L(k,k) = 1.0;
    
    % fix (1,1) ensure the well-conditioness of the matrix
    k = toIdx(1,1);
    L(k,:) = 0;
    L(k,k) = 1.0;
end

