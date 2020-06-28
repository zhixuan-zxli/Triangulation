function visEmbeddedGrid(filename)
    h = fopen(filename, 'rb');
    [l,b,dom] = readEmbeddedGrid(h);
    fclose(h);   
    disp(dom);    
    figure; imagesc(l{1}); title('Cell labels'); axis equal;
    figure; imagesc(l{2}); title('u labels'); axis equal;
    figure; imagesc(l{3}); title('v labels'); axis equal;
    figure; visCutCellBrdy(b);
end