function kn = linearizeSpline(pp, subdiv)
    if nargin == 1 || subdiv == 1
        newbrks = pp.breaks;
    else
        oldbrks = pp.breaks;
        n = numel(oldbrks);
        dt = diff(oldbrks);
        newbrks = zeros(subdiv,n-1);
        newbrks(1,:) = oldbrks(1:end-1);
        for i=1:subdiv-1
            newbrks(i+1,:) = newbrks(1,:) + i/subdiv*dt;
        end
        newbrks = [newbrks(:)' oldbrks(end)];
    end
    kn = fnval(pp, newbrks);
end