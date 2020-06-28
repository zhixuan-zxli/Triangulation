function pp = readpp(handle)
    tmp = fread(handle, 3, 'int');
    breaks = fread(handle, [tmp(3)+1 1], 'double');
    coefs = fread(handle, [tmp(2) tmp(1)*tmp(3)], 'double');
    coefs = coefs';
    coefs = coefs(:,end:-1:1);
    pp = ppmak(breaks,coefs,tmp(1));
end