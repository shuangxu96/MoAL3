% function [OutU,OutV,OutW,OutT,t]=UpdateM2(Tensor,InU,InV,InW,MaxIt,tol)
function [OutU,OutV,OutW,t]=UpdateM2(Tensor,W,InU,InV,InW,MaxIt,tol)

[d1 d2 d3]=size(Tensor);

[r]=size(InU,2);

t = sum(sum(sum((W.*Tensor).^2)));

for iter=1:MaxIt
    %iter
    OutW=UpdateM2w(Tensor,W,InU,InV);      %%%均为元胞数组形式
    OutV=UpdateM2v(Tensor,W,InU,OutW);
    OutU=UpdateM2u(Tensor,W,OutV,OutW);
    
    Tensor_dec=zeros(d1,d2,d3);
    for jj=1:r
        uv=OutU(:,jj)*OutV(:,jj)';
        uv1=repmat(uv,1,d3);
        for i=1:d3         %%%size(w1)=d1*(d2*d3)
            w0(:,:,i)=repmat(OutW(i,jj)',d1,d2);
        end
        w1=reshape(w0,d1,d2*d3);
        uvw1=uv1.*w1;
        uvw(:,:,:,jj)=reshape(uvw1,d1,d2,d3);    %%%%%%张量形式  !!!!!!!
        Tensor_dec=Tensor_dec + uvw(:,:,:,jj);
    end
    t = [t sum(sum(sum((W.*(Tensor-Tensor_dec)).^2)))];
    sum(sum(sum((W.*(Tensor-Tensor_dec))))/sum(sum(sum((W.*Tensor).^2))));
    
%     disp(['Relative reconstruction error ', num2str(sum(sum(sum((W.*(Tensor-Tensor_dec)).^2)))/sum(sum(sum((W.*Tensor).^2))))]);
%     disp(['norm u ', num2str(norm(InV - OutV))]);
%     disp(['norm v ', num2str(norm(InU - OutU))]);
    if norm(InV - OutV) < tol & norm(InU - OutU) < tol
        break;
    else
        InV = OutV;
        InU = OutU;
    end
    
end

% Nu = sqrt(sum(OutU.^2))';
% Nv = sqrt(sum(OutV.^2))';
% Nw = sqrt(sum(OutW.^2))';
% 
% No = diag(Nu.*Nv.*Nw);
% OutU = OutU*diag(1./Nu)*sqrt(No);
% OutV = OutV*diag(1./Nv)*sqrt(No);
% OutW= OutW*diag(1./Nw)*sqrt(No);



% Tensor_dec=zeros(d1,d2,d3);
% for jj=1:r
%     uv=OutU(:,jj)*OutV(:,jj)';
%     uv1=repmat(uv,1,d3);
%     for i=1:d3         %%%size(w1)=d1*(d2*d3)
%         w0(:,:,i)=repmat(OutW(i,jj)',d1,d2);
%     end
%     w1=reshape(w0,d1,d2*d3);
%     uvw1=uv1.*w1;
%     uvw(:,:,:,jj)=reshape(uvw1,d1,d2,d3);    %%%%%%张量形式  !!!!!!!
%     Tensor_dec=Tensor_dec + uvw(:,:,:,jj);
% end
% OutT=Tensor_dec;
