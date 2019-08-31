function [Br_trn,Bc_trn,Ar_trn,Ac_trn]=...
    Solver_optim_v1_entry(A_,Observed_Flag,Fr_trn,Fc_trn,latent_dim, Parameters, max_iter)
%% iteratively sovle: by ALS
% Optimizaition form ||A-Ar*Ac'||  + ||Ar-FrBr|| + ||Ac-FcBc||   + ||Ar||+
% ||Ac|| + ||Br|| + ||Bc||
% NOTE: some entries in A is not observed
% Observed_Flag has the same size as that of A  
% Observed_Flag(i,j)=1, observed, otherwise unobserved or tested

% DEMO:
% Observed_Flag=ones(size(DTI));
% idx_tst = randperm(numel(Observed_Flag), fix(numel(Observed_Flag)/10));
% Observed_Flag(idx_tst)=0;
% DTI_tst = DTI; DTI_tst(idx_tst)=0;
% [Br_trn,Bc_trn,Ar_trn,Ac_trn] = Solver_optim_v1_entry(DTI_tst,Observed_Flag,Fd,Ft,500);
% PredictedDTI = Ar_trn * Ac_trn';
% EstimationAUC(PredictedDTI(DTI==1 & Observed_Flag==0),PredictedDTI(DTI==0 & Observed_Flag==0),2000,0,true,true);
% DATE: 2017-11-08
assert( sum( abs( size(A_) - size(Observed_Flag) ) ) ==0 )

if nargin<7
    max_iter = 100;
end

% 1- Initialization
r = latent_dim; %fix( rank(A_trn) / 3 );
[U_adj,S_adj,V_adj]= svd(A_);
[U_adj,S_adj,V_adj,~]=CleanSVD(U_adj,S_adj,V_adj);
Ar_trn = U_adj*sqrt(S_adj); Ar_trn = Ar_trn(:,1:r);
Ac_trn = V_adj*sqrt(S_adj); Ac_trn = Ac_trn(:,1:r);


nComp  = 1;
[~,~,~,~,Br_trn]=plsregress( Fr_trn,Ar_trn, nComp);
[~,~,~,~,Bc_trn]=plsregress( Fc_trn,Ac_trn, nComp);

%2 - preparion and lamdaTODO: solve Fr and Fc
    Tr_trn = [ones(size(Fr_trn,1),1)  ,  Fr_trn ];
    Tc_trn = [ones(size(Fc_trn,1),1)  ,  Fc_trn ];
    
    lamda_ar = Parameters.lamda_ar; % for l2-norm: ||Ar||, default 0.5(S3), 0.05 (S2)
    lamda_ac = Parameters.lamda_ac;% for l2-norm: ||Ac||,default 0.5(S3), 0.05 (S2)
    lamda_r = Parameters.lamda_r; %default 0.05 (S3), 0.5 (S2)
    lamda_c = Parameters.lamda_c; %default 0.05 (S3), 0.5 (S2)    
    
    tol_ = 1e-4;


% 3- solve optimization by ALS
error(1)=  ( norm(A_-Ar_trn*Ac_trn','fro') +   norm(Ar_trn-Tr_trn*Br_trn,'fro') +  norm(Ac_trn-Tc_trn*Bc_trn,'fro') ) /3 ;
for it_ = 1: max_iter
%     fprintf(1,'iter= %d-CV \n',it_);
    %(1) sovle Ar, by fixing other five matrices;
    % sovle: 2*Ar*Ac'*Ac-2* A *Ac+2*Ar-2*([ones(size(Fr,1),1)  ,  Fr ] * Br) =0;
    % =>Ar * Ac' * Ac +Ar  - A *Ac -[ones(size(Fr,1),1)  ,  Fr ] * Br =0;
    % =>Ar ( Ac' * Ac +I)  = A *Ac+[ones(size(Fr,1),1)  ,  Fr ] * Br ;
    %     xA=B ==> A'x' =B': x'= A'\B'
%     Ar_trn = (  ( Ac_trn' * Ac_trn +eye(size(Ac_trn,2)) )' \  ( A_trn *Ac_trn+Tr_trn * Br_trn )'  )';  
    % extende by adding regularization l2-norm of Ar
    %  Fully observed matrix has a close form solution.
%     Ar_trn = (  ( Ac_trn' * Ac_trn +(lamda_ar+1)*eye(size(Ac_trn,2)) )' \  ( A_ *Ac_trn+Tr_trn * Br_trn )'  )';  
        % element-wise
        Ar_trn = UpdateElement(A_,Observed_Flag,Ar_trn,Ac_trn,Tr_trn,Br_trn,lamda_ar);
    % approimate close-form solution,  not good enough but fast
%      Ar_trn = (  ( Ac_trn' * Ac_trn +(lamda_ar+1)*eye(size(Ac_trn,2)) )' \  ( (Observed_Flag.*A_) *Ac_trn+Tr_trn * Br_trn )'  )';  
       
    %(2) sovle Ac, by fixing other five matrices;
    % sovle: 2*Ac*Ar'*Ar-2* A' *Ar+2*Ac-2*([ones(size(Fc,1),1)  ,  Fc ] * Bc) =0;
    % => Ac*Ar'*Ar- A' *Ar+Ac-([ones(size(Fc,1),1)  ,  Fc ] * Bc) =0;
    % => Ac(Ar'*Ar+I) =  A' *Ar+([ones(size(Fc,1),1)  ,  Fc ] * Bc) ;
%     Ac_trn = ( (Ar_trn'*Ar_trn+eye(size(Ar_trn,2)) )' \  (A_trn' *Ar_trn+Tc_trn * Bc_trn )' )';
    % extende by adding regularization l2-norm of Ac
    % fully observed
%     Ac_trn = ( (Ar_trn'*Ar_trn+(lamda_ac+1)*eye(size(Ar_trn,2)) )' \  (A_' *Ar_trn+Tc_trn * Bc_trn )' )';
    % element-wise 
    Ac_trn = UpdateElement(A_',Observed_Flag',Ac_trn,Ar_trn,Tc_trn,Bc_trn,lamda_ac);
    
    % approimate close-form solution, not good enough but fast
%     Ac_trn = ( (Ar_trn'*Ar_trn+(lamda_ac+1)*eye(size(Ar_trn,2)) )' \  ((Observed_Flag.*A_)' *Ar_trn+Tc_trn * Bc_trn )' )';
    
    %(3)  sovle Br, by fixing other five matrices; 
    % solve: -2*Fr'*Ar + 2*Fr' * Fr * Br =0; No regularization
    %  (Fr' * Fr) * Br =Fr'*Ar 
    % solve: -2*Fr'*Ar + 2*Fr' * Fr * Br + lamda_r*diff(norm(Br,2) ) =0;  regularization
             Num_r = Tr_trn' * Tr_trn + lamda_r*eye(size(Tr_trn,2));  % L2-norm, fast convergence
            Den_r = Tr_trn'*Ar_trn;
    
        if det(Num_r) < 1e-6 % for singular matrix
            Br_trn = pinv(Num_r)* (Den_r) ;
        else
            Br_trn = Num_r \ Den_r ;
        end

    
    %(4)  sovle Bc, by fixing other five matrices
    %solve: -2*Fc'*Ac + 2*Fc' * Fc * Bc =0; No regularization
    %solve: -2*Fc'*Ac + 2*Fc' * Fc * Bc+ lamda_c*diff(norm(Bc,2) ) =0;  regularization
            Num_c = Tc_trn' * Tc_trn +  lamda_c*eye(size(Tc_trn,2)); % L2-norm, fast convergence
            Den_c = Tc_trn'*Ac_trn;
    
         if det(Num_c) < tol_% for singular matrix : + eye(size(Tc_trn,2)) regularization
            Bc_trn =  pinv( Num_c )* Den_c ;
        else
            Bc_trn =  Num_c \ Den_c ;
        end

    
    error(it_+1) =  ( norm(A_-Ar_trn*Ac_trn','fro') +   norm(Ar_trn-Tr_trn*Br_trn,'fro') +  norm(Ac_trn-Tc_trn*Bc_trn,'fro') ) /3 ;
    if mod(it_,10)==0
        fprintf(1,'%d: %.3f \n', it_, error(it_+1));
    end
    if abs(error(it_)- error(it_+1) )<= tol_
        disp('Converge')
        break;
    end

end

%%
function U_ = UpdateElement(A_,Observed_Flag,U_,V_,F_U,B_U,lambda)
% A= U * V', U= F * B, ||U||, ||V||, ||B||
Observed_Flag(Observed_Flag~=1) =0;

[M,N]=size(Observed_Flag);
[~,K]=size(U_);

% % vector form faster
for m=1:M
    w = Observed_Flag(m,:);
    idx_obs = find(w==1); % observed
    num_ = (A_(m,idx_obs) - U_(m,:) * V_(idx_obs,:)' ) * V_(idx_obs,:) + F_U(m,:) * B_U(:, :) ;
    den_ = ( (V_(idx_obs,:)' * V_(idx_obs,:))  )+ lambda*eye( size(V_,2) ) + eye( size(V_,2) );
    
    if sum( isnan(den_(:)) ) || sum( isinf(den_(:)) )
        error (den_)
    end
    U_(m,:) = num_ / den_; %(num_ * den_')  / (den_ * den_');
end

% % element form fast
% for m=1:M
%     w = Observed_Flag(m,:);
%     idx_obs = find(w==1); % observed
%     for k=1:K
%         % for U_m_k
%         den_ = ( V_(idx_obs,k)' *V_(idx_obs,k) +1+1); %
%         num_1 = F_U(m,:) * B_U(:, k) ;
%         num_2=( A_(m,idx_obs) - U_(m,:) * V_(idx_obs,:)'  ) * V_(idx_obs,k);
%         U_(m,k) =  (num_1+num_2)/den_;
%    
%     end
%   
% end

%  % element form slower
% for m=1:M
%     for k=1:K
%         % for U_m_k
%         den_ = 1 +1+sum( V_(:,k) .* V_(:,k) ); % wrong here
%         num_ = F_U(m,:) * B_U(:, k) ;
%      
%         for n=1: N
%             if Observed_Flag(m,n)==1 %#% observed %#%
%                 temp_ = A_(m,n) - ( U_(m,:) * V_(n,:)' - U_(m,k) * V_(n,k) ); % remove itself
%                 num_ = num_ +  temp_* V_(n,k);
%             end
%         end
%         
%         U_(m,k) = num_/den_;
%     end
%     ;
% end


%%
function [U_c,S_c,V_c,The_end]=CleanSVD(U,S,V)
% Remove small SVs

svd_entry = diag(S);
small_idx = find(svd_entry<1e-6) ;
if isempty(small_idx)
    The_end = length(svd_entry);
else
    The_end = small_idx(1)-1;
end


U_c= U(:,1:The_end);
S_c = S(1:The_end,1:The_end);
V_c = V(:,1:The_end);
