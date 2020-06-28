function a = signedArea(verts)
% verts(end,:) ~= verts(1,:)
    nv = size(verts,1);
    a = 0;
    for i=2:nv-1
        v1 = verts(i,:) - verts(1,:);
        v2 = verts(i+1,:) - verts(1,:);
        a = a + 0.5*(v1(1)*v2(2)-v1(2)*v2(1));
    end
end