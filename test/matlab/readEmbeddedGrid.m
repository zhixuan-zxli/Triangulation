function [cellLabels, cellBrdy, rectdomain] = readEmbeddedGrid(handle)
    rectdomain.size = fread(handle, [1 2], 'int');
%     rectdomain.origin = fread(handle, [1 2], 'double');
    rectdomain.dx = fread(handle, [1 2], 'double');
    cellLabels = cell(1, 3);
    cellLabels{1} = fread(handle, rectdomain.size, 'int');
    cellLabels{2} = fread(handle, rectdomain.size + [1 0], 'int');
    cellLabels{3} = fread(handle, rectdomain.size + [0 1], 'int');
    numCurves = fread(handle, 1, 'int');
    cellBrdy = cell(numCurves, 1);
    for i=1:numCurves
        cellBrdy{i} = readpp(handle);
    end
end