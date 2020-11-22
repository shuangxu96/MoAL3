function C=UpdateM2u(Tensor,W,A,B)
r=size(A,2);
[m n p]=size(Tensor);

W=permute(W,[2,3,1]);
Tensor=permute(Tensor,[2,3,1]);

for j=1:r              
    U1=A(:,j)*B(:,j)';
    U2=reshape(repmat(U1,1,m),n,p,m);
    U(:,:,:,j)=W.*U2;     
end


for ii=1:m
    for kk=1:r
        M1(:,kk) = vec(U(:,:,ii,kk));
    end
    M(:,:,ii)=M1;
  
end
for i=1:m
    M_cell(1,i)=mat2cell(M(:,:,i),size(A,1)*size(B,1),r);
    
    TX=W.*Tensor;
    b=vec(TX(:,:,i));
    B_cell(1,i)=mat2cell(b,size(A,1)*size(B,1),1);
end

C_cell=cellfun(@MSE2,M_cell,B_cell,'UniformOutput',false);  %A*C=B matrix form

C=cell2mat(C_cell)';