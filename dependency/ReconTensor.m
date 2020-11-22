function X = ReconTensor(U,V,T)
[m,n,p,r] = deal(size(U,1),size(V,1),size(T,1),size(T,2));
X = zeros(m,n,p);
for idx = 1:r
uv = U(:,idx)*V(:,idx)';
uv1 = repmat(uv,1,p);
for idy = 1:p
w0(:,:,idy) = repmat(T(idy,idx)',m,n);
end
w1 = reshape(w0,m,n*p);
uvt1 = uv1.*w1;
uvt(:,:,:,idx) = reshape(uvt1,m,n,p);
X = X+uvt(:,:,:,idx);
end
end