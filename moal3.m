function [OutU, OutV, OutT, OutW, MoAL, Label, llh] = moal3(W, X, r, varargin)
%moal3 Perform low-rank tensor factorization with mixture of asymmetric
%Laplacian (MoAL) noise.
%
%   Positional parameters:
%
%     W                An binary matrix indicating whether the entry is missing
%     X                An observed matrix
%     r                Rank
%   
%   Optional input parameters:  
%
%     'MaxIter'             The maximum number of outer iterations.(Defualt: 30)
%     'K'                   The number of asymmetric Laplacian distributions. (Defualt: 4)
%     'NumIter'             The maximum number of inner iterations in M-step. (Defualt: 100)
%     'TuningK'             Binary scalar. Code automatically tunes K if it
%                           takes true, and does not tune otherwise. (Defualt: true)
%     'IniU','IniV','IniT'  Initial U and V.
%     'tol'                 Converge tolerence. (Defualt: 1e-5)
%
%   Return values:
%     OutU, OutV, OutT Final estimate of U, V and T.
%     OutW             The learned weights for each entry.
%     Label            The class label for each entry.
%     llh              The likelihood of each iteration.
%     MoAL             MoAL is a struct that contains information of 
%                      mixture of asymmetric Laplacian distribution. MoAL
%                      constains following fields:
%
%       'alpha'        The location parameter
%       'lambda'       The scale parameter                          
%       'kappa'        The skew parameter                          
%       'pi'           The mixing proportion parameter  

%   References: [1] Huang X F, Xu S, Zhang C X, Zhang J S. Robust CP Tensor
%   Factorization With Skew Noise. IEEE Signal Processing Letters, vol. 27,
%   pp. 785-789, December 2020. 
%   [2] Xu S, Zhang C X, Zhang J S. Adaptive
%   Quantile Low-Rank Matrix Factorization. Pattern Recognition, vol. 103,
%   article number 107310, July 2020.

%   Written by Shuang Xu (xu.s@outlook.com; shuangxu@stu.xjtu.edu.cn).

% =========================
% parse the input arguments
% =========================
index = find(W(:)~=0);
if nargin > 2
    [MaxIter, K, NumIter, TuningK, IniU, IniV, IniT, tol] = parseArgs_moal3(varargin, X, index, r);
end

% =========================
% Initialize the class label
% =========================
R = Initialize_moal3(X(index)',K);
[~,Label(1,:)] = max(R,[],2);
R = R(:,unique(Label)); % delete blank class

% =========================
% Initialize MoAL parameters
% =========================
MoAL.alpha  = zeros(1,K);
MoAL.lambda = rand(1,K);
MoAL.kappa  = rand(1,K);
MoAL.pi     = mean(R,1);

% =========================
% Initial E-step
% =========================
[TempU, TempV, TempT] = deal(IniU,IniV,IniT);
TempX = ReconTensor(TempU,TempV,TempT);
Error = X(:) - TempX(:);
Error = Error(index);
t = 1;
[R, llh(t)] = Estep_moal3(Error', MoAL);

% =========================
% main loop
% =========================
flag = false;
while ~flag
    t = t+1;
    [old_U,old_V] = deal(TempU,TempV);
    
    % M-E-M-E step
    [MoAL, rho]           = Mstep_Model_moal3(Error, R, MoAL);
    [R, llh(t)]           = Estep_moal3(Error', MoAL);
    [TempW, TempU, TempV, TempT] = Mstep_UV_moal3(W,X,TempU,TempV,TempT,R,NumIter,rho,MoAL);
    TempX                 = ReconTensor(TempU,TempV,TempT);
    Error                 = X(:)-TempX(:);
    Error                 = Error(index);
    [R, llh(t)]           = Estep_moal3(Error', MoAL);
    
    % Tuning the number of ALDs, K
    old_K = K;
    if TuningK %&& t>5
        [~,Label(:)] = max(R,[],2);
        u = unique(Label);   % non-empty components
        if size(R,2) ~= size(u,2)
            R = R(:,u);   % remove empty components
            MoAL.alpha  = MoAL.alpha(:,u);
            MoAL.lambda = MoAL.lambda(:,u);
            MoAL.kappa  = MoAL.kappa(:,u);
            MoAL.pi     = MoAL.pi(:,u);
            K = length(u);
            fprintf('Iteration %d: The number of ALDs (K) is reduced to %d.\n',t, K)
        end
        
    end
    
    % Converge or not
    if t>=MaxIter
        break
    end
    if K~=old_K
        flag = false;
    elseif sum(MoAL.pi)~=1
        flag = false;
    else
        flag = max(norm(old_U-TempU),norm(old_V-TempV))<tol;
    end
    
    % display the iteration number
    if numel(X)>500^2
        fprintf('Iteration %d \n', t)
    end
end
fprintf('Iteration %d: Algorithm stops.\n', t)
OutU = TempU;
OutV = TempV;
OutT = TempT;
OutW = TempW;
end

%% subfunctions - Mstep_UV_moal3
function [TempW, TempU, TempV, TempT] = Mstep_UV_moal3(W,X,TempU,TempV,TempT,R,NumIter,rho,MoAL)
TempW = zeros(size(X));
lambda = MoAL.lambda;

TempW(W(:)~=0) = sum(R.*rho.*lambda,2);
TempW = TempW./ (abs(X-ReconTensor(TempU,TempV,TempT))+1e-10); %for numerical stability

[TempU,TempV,TempT] =UpdateM2(X,sqrt(TempW),TempU,TempV,TempT,NumIter,0.00000001);
end

%% subfunctions - Mstep_Model_moal3
function [MoAL,rho] = Mstep_Model_moal3(Error,R,MoAL)
kappa = MoAL.kappa;

% pi
nk = sum(R,1);
pi = nk/size(R,1);

% lambda
sign = Error<0;
rho = sign*(1-kappa) + ~sign*kappa;
lambda = nk./ sum(R.*rho.*abs(Error));
indINF = find(~isfinite(lambda));
if ~isempty(indINF)
    lambda(indINF) = nk(indINF)/1e-6;
end
% kappa
eta = sum(R.*Error).*lambda;
sDelta = sqrt(4*nk.^2+eta.^2);
kappa = (2*nk + eta - sDelta) ./ (2*eta);

MoAL.lambda = lambda;
MoAL.kappa  = kappa;
MoAL.pi     = pi;
end

%% subfunctions - Estep_moal3
function [R, llh] = Estep_moal3(Error, MoAL)
% Compute the conditional probability
alpha  = MoAL.alpha;
lambda = MoAL.lambda;
kappa  = MoAL.kappa;
pi     = MoAL.pi;

n = size(Error,2);
k = size(alpha,2);
logRho = zeros(n,k);

for i = 1:k
    logRho(:,i) = logpdfald(Error,alpha(i),lambda(i),kappa(i));
end
logRho = bsxfun(@plus,logRho,log(pi));
T = logsumexp(logRho,2);
llh = sum(T)/n; % loglikelihood
logR = bsxfun(@minus,logRho,T);
R = exp(logR);
end

%% subfunctions - initialize_moal3
function R = Initialize_moal3(X, k)
[~,n] = size(X);
idx = randsample(n,k);
m = X(:,idx);
[~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
[u,~,label] = unique(label);
while k ~= length(u)
    idx = randsample(n,k);
    m = X(:,idx);
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    [u,~,label] = unique(label);
end
R = full(sparse(1:n,label,1,n,k,n));
end

%% subfunctions - parseArgs_moal3
function [MaxIter, K, NumIter, TuningK, IniU, IniV, IniT, tol] = parseArgs_moal3(vararginCell, X, index, r)
[vararginCell{:}] = convertStringsToChars(vararginCell{:});

if sum(strcmp(vararginCell, 'IniU'))==0
    IniU = InitializeComponent(X, index, size(X,1), r);
else
    IniU = [];
end
if sum(strcmp(vararginCell, 'IniV'))==0
    IniV = InitializeComponent(X, index, size(X,2), r);
else
    IniV = [];
end
if sum(strcmp(vararginCell, 'IniT'))==0
    IniT = InitializeComponent(X, index, size(X,3), r);
else
    IniT = [];
end

pnames = { 'MaxIter' 'K' 'NumIter' 'TuningK' 'IniU' 'IniV' 'IniT'  'tol'};
dflts  = {     20,   4,    20,      true,    IniU,   IniV,  IniT,  1e-50};
[MaxIter, K, NumIter, TuningK, IniU, IniV, IniT, tol] ...
    = internal.stats.parseArgs(pnames, dflts, vararginCell{:});
end

%% subfunctions - InitializeComponent
function IniU = InitializeComponent(X, index, shape1, shape2)
s = median(abs(X(index)));
s = sqrt(s/shape2);
if min(X(index)) >= 0
    IniU = rand(shape1, shape2)*s;
else
    IniU = rand(shape1, shape2)*s*2-s;
end
end

%% subfunctions - logsumexp
function s = logsumexp(x, dim)
% Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
%   By default dim = 1 (columns).
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 1
    % Determine which dimension sum will use
    dim = find(size(x)~=1,1);
    if isempty(dim), dim = 1; end
end

% subtract the largest in each column
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
i = find(~isfinite(y));
if ~isempty(i)
    s(i) = y(i);
end
end

%% subfunctions - logpdfald
function logy = logpdfald(x,alpha,lambda,kappa)
x = x(:);
xa = x-alpha;
xabs = abs(xa);
sign = (xa >= 0);
al_kernal = lambda*xabs.*( kappa*sign + (1-kappa)*~sign );
logy = log(lambda*kappa*(1-kappa)) - al_kernal;
end