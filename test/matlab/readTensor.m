function ts = readTensor(hd, dtype)
    if nargin == 1
        dtype = 'double';
    end
    D = fread(hd, 1, 'int');
    sz = fread(hd, [1 D], 'int');
    ts = fread(hd, sz, dtype);
end