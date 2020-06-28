function L = genLaplacianCvl(H, dx, bctype)
%genLaplacianCvl : Generate the Laplacian matrix on the curvilinear mesh
%whose metric is defined by H. 
% H{1}, H{2} : metric on the staggered faces.
% H{3} : metric at the cell centers.
    N = size(H{3});
    numUnk = (N(1)+2) * (N(2)+2);
    toIdx = @(i,j) j .* (N(1)+2) + i + 1;
    L = zeros(numUnk);
    for j=1:N(2)
        for i=1:N(1)
            k = toIdx(i, j);
            g_sqr = H{3}(i,j,1) * H{3}(i,j,2);
            % dimension 1
            L(k,k-1) = 1.0 / g_sqr * (H{1}(i,j,2) / H{1}(i,j,1)) / (dx(1).^2);
            L(k,k+1) = 1.0 / g_sqr * (H{1}(i+1,j,2) / H{1}(i+1,j,1)) / (dx(1).^2);
            L(k,k) = L(k,k) - L(k,k-1) - L(k,k+1);
            % dimension 2
            L(k,k-(N(1)+2)) = 1.0 / g_sqr * (H{2}(i,j,1) / H{2}(i,j,2)) / (dx(2).^2);
            L(k,k+(N(1)+2)) = 1.0 / g_sqr * (H{2}(i,j+1,1) / H{2}(i,j+1,2)) / (dx(2).^2);
            L(k,k) = L(k,k) - L(k,k-(N(1)+2)) - L(k,k+(N(1)+2));
        end
    end
    
    if strcmp(bctype,'D')
        % low face of dimension 1
        k = toIdx(0, 0:N(2)+1);
        % high face of dimension 1
        k = [k toIdx(N(1)+1, 0:N(2)+1)];
        % low face of dimension 2
        k = [k toIdx(0:N(1)+1,0)];
        % high face of dimension 2
        k = [k toIdx(0:N(1)+1,N(2)+1)];
        %
        for u=k, L(u,u) = 1.0; end
    elseif strcmp(bctype,'N')
        % lo, hi faces of dimension 1
        for j=1:N(2)
            k = toIdx(0,j);
            L(k,k) = 1.0 / (H{1}(1,j,1) * dx(1));
            L(k,k+1) = -1.0 / (H{1}(1,j,1) * dx(1));
            k = toIdx(N(1)+1,j);
            L(k,k) = 1.0 / (H{1}(N(1)+1,j,1) * dx(1));
            L(k,k-1) = -1.0 / (H{1}(N(1)+1,j,1) * dx(1));
        end
        % lo, hi faces of dimension 2
        for i=1:N(1)
            k = toIdx(i,0);
            L(k,k) = 1.0 / (H{2}(i,1,2) * dx(2));
            L(k,k+(N(1)+2)) = -1.0 / (H{2}(i,1,2) * dx(2));
            k = toIdx(i,N(2)+1);
            L(k,k) = 1.0 / (H{2}(i,N(2)+1,2) * dx(2));
            L(k,k-(N(1)+2)) = -1.0 / (H{2}(i,N(2)+1,2) * dx(2));
        end
        % the four corner points
        k = toIdx(0,0);
        L(k,k) = 1.0;
        k = toIdx(N(1)+1,0);
        L(k,k) = 1.0;
        k = toIdx(0, N(2)+1);
        L(k,k) = 1.0;
        k = toIdx(N(1)+1, N(2)+1);
        L(k,k) = 1.0;
        % fix (1,1) to retain a well-conditioned matrix
        k = toIdx(1,1);
        L(k,:) = 0;
        L(k,k) = 1.0;
    end
end

