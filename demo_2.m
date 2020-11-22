clear;
clc;

I=[50 50 50];
r=5;
N=numel(I);
for kk=1:30
    
    X_Ori = ReconTensor(randn(I(1),r), randn(I(2),r), randn(I(3),r));
    
    m=I(1);
    n=I(2);
    p=I(3);
    
    Ind = randperm(m*n*p);
    
    p1 = floor(m*n*p*0.2);  %missing percentage
    W = ones(m,n,p);
    W(Ind(1:p1)) = 0;       %Indicator matrix
    
    %% mixture noise 
    noi_type = 1;
    if noi_type == 1
        noi = randald(m*n*p, 0, 25, 0.5);
        noi = reshape(noi, [m,n,p]);
        X_Noi = W.*(X_Ori+noi);
    elseif noi_type == 2
        noi = randald(m*n*p, 0, 25, 0.1);
        noi = reshape(noi, [m,n,p]);
        X_Noi = W.*(X_Ori+noi);
    elseif noi_type == 3
        noi(1:(m*n*p*0.2)) = randn(1,m*n*p*0.2)*0.15;
        noi((m*n*p*0.2+1):(m*n*p*0.4)) = randn(1,m*n*p*0.2)*0.1;
        noi((m*n*p*0.4+1):m*n*p) = randald(m*n*p*0.6, 0, 25, 0.1);
        noi = noi(randperm(m*n*p));
        noi = reshape(noi, [m,n,p]);
        X_Noi = W.*(X_Ori+noi);
    elseif noi_type == 4
        noi(1:(m*n*p*0.2)) = randald(m*n*p*0.2, 0, 25, 0.15);
        noi((m*n*p*0.2+1):(m*n*p*0.4)) = randald(m*n*p*0.2, 0, 25, 0.2);
        noi((m*n*p*0.4+1):m*n*p) = randald(m*n*p*0.6, 0, 25, 0.1);
        noi = noi(randperm(m*n*p));
        noi = reshape(noi, [m,n,p]);
        X_Noi = W.*(X_Ori+noi);
    end
        
    aa = median(abs(X_Noi(Ind(p1+1:end))));
    aa = sqrt(aa/r);
    
    U0 = rand(m,r)*aa*2-aa;
    V0 = rand(n,r)*aa*2-aa;
    W0 = rand(p,r)*aa*2-aa;
    UV0 = rand(m*n,r)*aa*2-aa;
    
    % MoAL3
    [OutU, OutV, OutT, OutW, MoAL, Label, llh] = moal3(W, X_Noi, r, 'MaxIter', 40, 'IniU', U0, 'IniV', V0, 'IniT', W0, 'K', 3, 'NumIter', 40, 'tol', 1e-50, 'TuningK', true);
    E1 = X_Ori - ReconTensor(OutU, OutV, OutT);
    
    E1G(kk) = mean(abs(E1(:)));
end

mean([E1G',E2G',E3G',E4G',E5G'])
median([E1G',E2G',E3G',E4G',E5G'])