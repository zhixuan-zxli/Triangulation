function visCutCellBrdy(cutCellBrdy, labels, rdom)
    subdiv = 1;
    if cutCellBrdy{1}.order > 2, subdiv = 4; end
    
    nb = numel(cutCellBrdy);
    hold on;
    % plot the irregular cells first
    for j=1:nb
        vtx = linearizeSpline(cutCellBrdy{j}, subdiv);
        if signedArea(vtx') > 0
            h = fill(vtx(1,:), vtx(2,:), 'm');
        else
            h = fill(vtx(1,:), vtx(2,:), 'w');
        end
        h.PickableParts = 'none';
    end
    % fill the regular cells
    if nargin > 1
        [row,col] = find(labels == 1);
        for j=1:numel(row)
            p = rdom.origin + rdom.dx .* [row(j)-1 col(j)-1];
            h = fill([p(1) p(1)+rdom.dx(1) p(1)+rdom.dx(1) p(1)], ...
                [p(2) p(2) p(2)+rdom.dx(2) p(2)+rdom.dx(2)], 'm');
            h.PickableParts = 'none';
        end
    end
    hold off;
    axis equal;
end