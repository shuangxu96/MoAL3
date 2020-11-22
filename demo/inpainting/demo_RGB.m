clear;
clc;
close all;

img_num=6

X_Ori=im2double(imread(['re',num2str(img_num),'.jpg']));   %%%原图 0-255  Lena.jpg

[m n p]=size(X_Ori);
% W = zeros(m,n,p);
r=30;

% Ind = randperm(m*n*p);
% Ind = Ind(1:ceil(.8*numel(Ind)));
% W(Ind) = 1;
X_Noi = im2double(re6_1);
W = abs(X_Ori-X_Noi);
W(W>=0.1)=1;
W(W<0.1)=0;
W = mean(W,3);
W(W>=0.1)=1;
W(W<0.1)=0;
imshow(W)
W=repmat(W,1,1,3);
W=1-W;
X_Noi = X_Ori;       %Add missing components
X_Noi(W==0) = 1;

% save('W.mat','W')
save_path = ['.\RGB_result_img#',num2str(img_num)];
mkdir(save_path)
%% mixture noise [-5,5] N(0,0.2) N(0,0.1)
figure,
imshow(X_Noi)
        
        % X_t=tensor(X_Noi);
        % P = cp_als(X_t,r);
        % U0=P.U{1};
        % V0=P.U{2};
        % W0=P.U{3};
        aa = median(abs(X_Noi(W~=0)));
        aa = sqrt(aa/r);
        U0 = rand(m,r)*aa*2-aa;
        V0 = rand(n,r)*aa*2-aa;
        W0 = rand(p,r)*aa*2-aa;
        UV0 = rand(p*n,r)*aa*2-aa;
        % MoAL3
        [OutU, OutV, OutT, OutW, MoAL, Label, llh] = moal3(W, X_Noi, r, 'MaxIter', 15, 'IniU', U0, 'IniV', V0, 'IniT', W0, 'K', 2, 'NumIter', 20, 'tol', 1e-50);
        X_Rec = ReconTensor(OutU, OutV, OutT);
        imwrite(X_Rec, [save_path, '\moal3.png'])
        
        % MoAL2
        [OutU, OutV, OutW, MoAL, Label, llh] = moal2(reshape(W,m,n*p), reshape(X_Noi,m,n*p), r, 'MaxIter', 15, 'K', 2, 'NumIter', 50, 'tol', 1e-50);
        X_Rec = reshape(OutU*OutV',[m,n,p]);
        imwrite(X_Rec, [save_path, '\moal2.png'])
        
        % MoG3
        param.maxiter =15;
        param.OriX = X_Ori;
        param.InU = U0;
        param.InV = V0;
        param.InW = W0;
        param.k =2;
        param.display = 0;
        param.NumIter =20; %%15
        param.tol = 1.0e-50;
        param.method = 2;    %%%1-4选择L2MF求解方法，，1时间太长，2时间短为默认方法，3 报错 ERROR: Sigma is not PD，4 无此函数
        
        [label,model,TW,A,B,C,llh] =  MLGMDN3(W,X_Noi,r,param);  %%%EM算法
        X_Rec = ReconTensor(A, B, C);
        imwrite(X_Rec, [save_path, '\mog3.png'])
        
        % MoG2
        param.maxiter = 15;
        param.OriX = reshape(X_Noi, m,n*p);
        param.k = 2;
        param.InU = U0;
        param.InV = UV0;
        aram.display = 0;
        param.NumIter = 50;
        param.tol = 1.0e-50;
        param.method = 2;
        
        [label, model,TW, A,B,llh] =  MLGMDN(reshape(W,m,n*p), reshape(X_Noi,m,n*p),r,param);
        X_Rec = reshape(A*B',[m,n,p]);
        imwrite(X_Rec, [save_path, '\mog2.png'])
%         
%         % Parafac
%         [X_parafac, k] = PARAFAC(X_Noi);
%         imwrite(X_parafac, [save_path, '\parafac.png'])
%         
%         % CWM
%         param.maxiter=15;
%         [U,V] = CWMmc(reshape(X_Ori,m,n*p),reshape(W,m,n*p),r,param);
%         imwrite(reshape(U*V',[m,n,p]) , [save_path, '\cwm.png'])
%         
    [P0,P1] = cp_wopt(tensor(X_Noi),W,r);
    U0=P0.U{1};
    V0=P0.U{2};
    W0=P0.U{3};
    z=ktensor(P0);
    imwrite( double(z), [save_path,'\cp.png'])

        E1 = psnr(im2double(imread([save_path, '\moal3.png'])), X_Ori);
        S1 = ssim(im2double(imread([save_path, '\moal3.png'])), X_Ori);
        E2 = psnr(im2double(imread([save_path, '\moal2.png'])), X_Ori);
        S2 = ssim(im2double(imread([save_path, '\moal2.png'])), X_Ori);
        E3 = psnr(im2double(imread([save_path, '\mog3.png'])), X_Ori);
        S3 = ssim(im2double(imread([save_path, '\mog3.png'])), X_Ori);
        E4 = psnr(im2double(imread([save_path, '\mog2.png'])), X_Ori);
        S4 = ssim(im2double(imread([save_path, '\mog2.png'])), X_Ori);
        E5 = psnr(im2double(imread([save_path, '\cp.png'])), X_Ori);
        S5 = ssim(im2double(imread([save_path, '\cp.png'])), X_Ori);
%         E5 = psnr(im2double(imread([save_path, '\parafac.png'])), X_Ori);
%         S5 = ssim(im2double(imread([save_path, '\parafac.png'])), X_Ori);
%         E6 = psnr(im2double(imread([save_path, '\cwm.png'])), X_Ori);
%         S6 = ssim(im2double(imread([save_path, '\cwm.png'])), X_Ori);
        
        P = [E1(:),E2(:),E3(:),E4(:),E5(:)]
        S = [S1(:),S2(:),S3(:),S4(:),S5(:)]
        
        save(['metric_img#',num2str(img_num),'.mat'], 'P', 'S')



a=X_Ori(W==0);
b=X_Ori(W==1);
color = [0.850,0.325,0.0980;0.929,0.694,0.125;0.494,0.184,0.556;
    0.466,0.674,0.188;0.301,0.745,0.933;0.635,0.0780,0.184;0,0.447,0.741];
figure('Position',[50,50,712,320])
subplot(1,2,1)
ci = 7;
x=a;
hold on
histogram(x,'BinMethod','fd','EdgeColor','auto','EdgeAlpha',0.3,...
    'Normalization','pdf','FaceColor',color(ci,:),'NumBins',80)
opt.display = 0;
    [alpha,lambda,kappa] = mleald(x,opt);
h = plotpdfald(alpha,lambda,kappa,0,1);
h.Color = color(ci,:);
xlim([0,1])
rectangle('Position',[0.03,3.8,0.1,0.08],'FaceColor',color(7,:),'EdgeColor',color(7,:))
text(0.15,3.84,'Missing Data','FontSize',10)

ci = 1;
x=b;
hold on
histogram(x,'BinMethod','fd','EdgeColor','auto','EdgeAlpha',0.3,...
    'Normalization','pdf','FaceColor',color(ci,:),'NumBins',80)
opt.display = 0;
    [alpha,lambda,kappa] = mleald(x,opt);
h = plotpdfald(alpha,lambda,kappa,0,1);
h.Color = color(ci,:);
xlim([0,1])
rectangle('Position',[0.03,3.6,0.1,0.08],'FaceColor',color(1,:),'EdgeColor',color(1,:))
text(0.15,3.64,'Non-Missing Data','FontSize',10)
% gauss
subplot(1,2,2)
ci = 7;
x=a;
hold on
histogram(x,'BinMethod','fd','EdgeColor','auto','EdgeAlpha',0.3,...
    'Normalization','pdf','FaceColor',color(ci,:),'NumBins',80)
mua = mean(a); sigmaa = std(a);
series = linspace(0,1,500);
y = pdf('Normal',series,mua,sigmaa);
h = plot(series,y,'LineWidth',2);
h.Color = color(ci,:);
xlim([0,1])
rectangle('Position',[0.03,3.8,0.1,0.08],'FaceColor',color(7,:),'EdgeColor',color(7,:))
text(0.15,3.84,'Missing Data','FontSize',10)

ci = 1;
x=b;
hold on
histogram(x,'BinMethod','fd','EdgeColor','auto','EdgeAlpha',0.3,...
    'Normalization','pdf','FaceColor',color(ci,:),'NumBins',80)
mub = mean(b); sigmab = std(b);
y = pdf('Normal',series,mub,sigmab);
h = plot(series,y,'LineWidth',2);
h.Color = color(ci,:);
xlim([0,1])
rectangle('Position',[0.03,3.6,0.1,0.08],'FaceColor',color(1,:),'EdgeColor',color(1,:))
text(0.15,3.64,'Non-Missing Data','FontSize',10)

saveas(gcf, 'InpaintPDF.eps','epsc')
