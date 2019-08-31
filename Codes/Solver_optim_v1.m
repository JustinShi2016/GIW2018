function [Br_trn,Bc_trn,Ar_trn,Ac_trn]=...
    Solver_optim_v1(A_trn,Fr_trn,Fc_trn,latent_dim, Parameters, max_iter)
%% iteratively solve the Optimizaition form by ALS
%% ||A-Ar*Ac'||  + ||Ar-FrBr|| + ||Ac-FcBc||   
%% + Lar*||Ar|| +Lac*||Ac|| +Lr*||Br|| +Lc*||Bc||
% DATE: 2017-10-28
% Parameters - a structure containing regularization coefficients
if nargin<6
    max_iter = 100;
end
if nargin<4 % for S2, which is the most frequent scenario
    Parameters.lamda_ar = 0.05;
    Parameters.lamda_ac = 0.05;
    Parameters.lamda_r = 0.5;
    Parameters.lamda_c = 0.5;
end

sparse_option = 'L2';
disp(['Regularization:',sparse_option])
% 1- Initialization
[U_adj,S_adj,V_adj]= svd(A_trn);
[U_adj,S_adj,V_adj,~]=CleanSVD(U_adj,S_adj,V_adj);

r = latent_dim; %fix( rank(A_trn) / 3 );
Ar_trn = U_adj*sqrt(S_adj); Ar_trn = Ar_trn(:,1:r);
Ac_trn = V_adj*sqrt(S_adj); Ac_trn = Ac_trn(:,1:r);

% since the optimization is dependent to nComp in S2/S3/S4, just set it with 1 to speed up running
nComp  =1; %rank(A_trn) ; 
[~,~,~,~,Br_trn]=plsregress( Fr_trn, Ar_trn, nComp);
[~,~,~,~,Bc_trn]=plsregress( Fc_trn, Ac_trn, nComp);

%2 - preparion and lamda
    Tr_trn = [ones(size(Fr_trn,1),1)  ,  Fr_trn ];
    Tc_trn = [ones(size(Fc_trn,1),1)  ,  Fc_trn ];
%     lamda_ar = 0.05; % for l2-norm: ||Ar||, default 0.5(S3), 0.05 (S2)
%     lamda_ac =0.05;% for l2-norm: ||Ac||,default 0.5(S3), 0.05 (S2)
%     lamda_r = 0.5; %default 0.05 (S3), 0.5 (S2)
%     lamda_c = 0.5; %default 0.05 (S3), 0.5 (S2)
    lamda_ar = Parameters.lamda_ar; % for l2-norm: ||Ar||, default 0.5(S3), 0.05 (S2)
    lamda_ac = Parameters.lamda_ac;% for l2-norm: ||Ac||,default 0.5(S3), 0.05 (S2)
    lamda_r = Parameters.lamda_r; %default 0.05 (S3), 0.5 (S2)
    lamda_c = Parameters.lamda_c; %default 0.05 (S3), 0.5 (S2)    
    tol_ = 1e-4;

% 3- solve optimization by ALS
error(1)=  ( norm(A_trn-Ar_trn*Ac_trn','fro') +   norm(Ar_trn-Tr_trn*Br_trn,'fro') +  norm(Ac_trn-Tc_trn*Bc_trn,'fro') ) /3 ;
for it_ = 1: max_iter
    %(1) sovle Ar, by fixing other five matrices;
    % sovle: 2*Ar*Ac'*Ac-2* A *Ac+2*Ar-2*([ones(size(Fr,1),1)  ,  Fr ] * Br) =0;
    % =>Ar * Ac' * Ac +Ar  - A *Ac -[ones(size(Fr,1),1)  ,  Fr ] * Br =0;
    % =>Ar ( Ac' * Ac +I)  = A *Ac+[ones(size(Fr,1),1)  ,  Fr ] * Br ;
    %     xA=B ==> A'x' =B': x'= A'\B'
%     Ar_trn = (  ( Ac_trn' * Ac_trn +eye(size(Ac_trn,2)) )' \  ( A_trn *Ac_trn+Tr_trn * Br_trn )'  )';  
    % extende by adding regularization l2-norm of Ar
    Ar_trn = (  ( Ac_trn' * Ac_trn +(lamda_ar+1)*eye(size(Ac_trn,2)) )' \  ( A_trn *Ac_trn+Tr_trn * Br_trn )'  )';  
    
    %(2) sovle Ac, by fixing other five matrices;
    % sovle: 2*Ac*Ar'*Ar-2* A' *Ar+2*Ac-2*([ones(size(Fc,1),1)  ,  Fc ] * Bc) =0;
    % => Ac*Ar'*Ar- A' *Ar+Ac-([ones(size(Fc,1),1)  ,  Fc ] * Bc) =0;
    % => Ac(Ar'*Ar+I) =  A' *Ar+([ones(size(Fc,1),1)  ,  Fc ] * Bc) ;
%     Ac_trn = ( (Ar_trn'*Ar_trn+eye(size(Ar_trn,2)) )' \  (A_trn' *Ar_trn+Tc_trn * Bc_trn )' )';
    % extende by adding regularization l2-norm of Ac
    Ac_trn = ( (Ar_trn'*Ar_trn+(lamda_ac+1)*eye(size(Ar_trn,2)) )' \  (A_trn' *Ar_trn+Tc_trn * Bc_trn )' )';

    %(3)  sovle Br, by fixing other five matrices; 
    % solve: -2*Fr'*Ar + 2*Fr' * Fr * Br =0; No regularization
    %  (Fr' * Fr) * Br =Fr'*Ar 
    % solve: -2*Fr'*Ar + 2*Fr' * Fr * Br + lamda_r*diff(norm(Br,2) ) =0;  regularization
    switch sparse_option
        case 'L2' % fast
            Num_r = Tr_trn' * Tr_trn + lamda_r*eye(size(Tr_trn,2));  % L2-norm, fast convergence
            Den_r = Tr_trn'*Ar_trn;
   end
    
    if ~strcmp(sparse_option, 'L1')        
        if det(Num_r) < 1e-6 % for singular matrix
            Br_trn = pinv(Num_r)* (Den_r) ;
        else
            Br_trn = Num_r \ Den_r ;
        end
    end
    
    %(4)  sovle Bc, by fixing other five matrices
    %solve: -2*Fc'*Ac + 2*Fc' * Fc * Bc =0; No regularization
    %solve: -2*Fc'*Ac + 2*Fc' * Fc * Bc+ lamda_c*diff(norm(Bc,2) ) =0;  regularization
    switch sparse_option
        case 'L2' % ridge regression
            Num_c = Tc_trn' * Tc_trn +  lamda_c*eye(size(Tc_trn,2)); % L2-norm, fast convergence
            Den_c = Tc_trn'*Ac_trn;
    end
    
    if ~strcmp(sparse_option, 'L1')
        if det(Num_c) < tol_% for singular matrix : + eye(size(Tc_trn,2)) regularization
            Bc_trn =  pinv( Num_c )* Den_c ;
        else
            Bc_trn =  Num_c \ Den_c ;
        end
    end
    
    %
    error(it_+1) =  ( norm(A_trn-Ar_trn*Ac_trn','fro') +   norm(Ar_trn-Tr_trn*Br_trn,'fro') +  norm(Ac_trn-Tc_trn*Bc_trn,'fro') ) /3 ;
    if mod(it_,20)==0
        fprintf(1,'%d: %.3f \n', it_, error(it_+1));
    end
    if abs(error(it_)- error(it_+1) )<= tol_
        disp('Converge')
        break;
    end

end

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
