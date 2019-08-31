function Predicted_LabelMat = PredictS2_Optim...
    (TrnLabelMat,U_row,S_row,ID_TrainRow, ID_TestRow, U_col,S_col,threshold, Parameters)
if nargin < 6 || isempty(threshold)
    threshold=fix(rank(TrnLabelMat)/3);
end
disp('solved by optimization')
% REMARK: TrnLabelMat has the same numbers of columns of the original matrix and fewer rows.
% ID_TrainRow, ID_TestRow are based on  the original matrix as well.

threshold =max(1, min( threshold, rank(TrnLabelMat)-1 )  );
disp(threshold)

TrnLabelMat_Cleaned = TrnLabelMat; % may contain all-zero columns which cause S4 prediction
% clear training matrix
degree = sum(TrnLabelMat_Cleaned,1);
idx_Removed_Col= find(degree <=0);
idx_Remaining_Col = find(degree >0);
TrnLabelMat_Cleaned(:,idx_Removed_Col)=[];

if ~isempty(idx_Removed_Col)
    disp('S4 found in S2/S3 CV');
end

%% Optimizaition form ||A-Ar*Ac|| + ||Ar-FrBr|| + ||Ac-FcBc||   + ||Sr-FrFr'|| + ||Sc-FcFc'|| 
% Initial value%
%symmetric
Fr_trn = U_row(ID_TrainRow,:)*sqrt(S_row); %(ID_TrainRow,:)
Fc_trn = U_col(idx_Remaining_Col,:)*sqrt(S_col); % no column with all-zero

A = TrnLabelMat_Cleaned;

%% run 
latent_dim = threshold; 
[Br,Bc,Ar_trn,Ac_trn]=Solver_optim_v1(A,Fr_trn,Fc_trn,latent_dim,Parameters, 500);

% % only row view: this part is different to S4 and S1 
Fr_tst =  U_row(ID_TestRow,:)*sqrt(S_row) ;
Fr_argu= [ones(size(Fr_tst,1),1)  ,  Fr_tst ];
if size(Fr_argu,2) ~= size(Br,1)
    warning('inner dim');
end
Y_predicted_row = Fr_argu *Br;% Ar_trn
% %column view
Y_trn_col = [ones(size(Fc_trn,1),1)  ,  Fc_trn ] * Bc; %%Ac_trn (default);  

% % BE careful: only for those testing columns having all ZEROS
Fc_tst = U_col(idx_Removed_Col,:)*sqrt(S_col);
Fc_argu =  [ones(size(Fc_tst,1),1)  ,  Fc_tst ];
Y_predicted_col_s4 = Fc_argu*Bc;%

%% 
% Predicted_LabelMat  =Y_predicted_row * Y_trn_col' ; 

Predicted_LabelMat = zeros( length(ID_TestRow), size(TrnLabelMat,2));
Predicted_LabelMat(:,idx_Remaining_Col) =Y_predicted_row * Y_trn_col' ;
Predicted_LabelMat(:,idx_Removed_Col) =Y_predicted_row * Y_predicted_col_s4';

function [U,S,V]=TurnSimBySVD(Sim)
[U,S,V]= svd(Sim);
[U,S,V,~]=CleanSVD(U,S,V);

%%

