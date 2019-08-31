function Predicted_LabelMat = PredictS4_Optim(...
                TrnLabelMat,...
                U_row,S_row, U_col,S_col, ...
                ID_TrainRow, ID_TrainCol,ID_TestRow, ID_TestCol,...
                threshold,Parameters)
% TrnLabelMat - the block of training label matrix in the adjacent matrix)
% {U_row,S_row}, {U_col,S_col} - the SVD decomposition (U and S) of two similairty
%                                                matrices, accouting for row nodes and column nodes respectively.
% {ID_TrainRow, ID_TrainCol},{ID_TestRow, ID_TestCol} - their indices in the original adjacent matrix
% Predicted_LabelMat - the output scores for testing label block.
% 
% REMARK: TrnLabelMat has fewer columns and fewer rows than the original matrix .
% ID_TrainRow, ID_TrainCol,ID_TestRow, ID_TestCol are based on  the original matrix as well.

if nargin < 10 || isempty(threshold)
    threshold=fix(rank(TrnLabelMat)/4);
end

threshold =max(1, min( threshold, rank(TrnLabelMat)-1 ) );
fprintf('Latent Dimension=%d\n',threshold)

TrnLabelMat_Cleaned = TrnLabelMat; % may contain all-zero rows/columns/both which cause worse matrix factorization
% clear training matrix
degree_d = sum(TrnLabelMat_Cleaned,2);
idx_Removed_Row= find(degree_d <=0);
idx_Remaining_Row= find(degree_d >0); % S2 or S4 occur
TrnLabelMat_Cleaned(idx_Removed_Row,:)=[];

degree_t = sum(TrnLabelMat_Cleaned,1);
idx_Removed_Col= find(degree_t <=0);  % S3 or S4 occur
idx_Remaining_Col = find(degree_t >0);
TrnLabelMat_Cleaned(:,idx_Removed_Col)=[];  

% run regression
A_trn = TrnLabelMat_Cleaned;
Fr_trn = U_row(ID_TrainRow(idx_Remaining_Row),:)*sqrt(S_row);
Fc_trn = U_col(ID_TrainCol(idx_Remaining_Col),:)*sqrt(S_col);

latent_dim=threshold; 
[Br,Bc,Ar,Ac]=Solver_optim_v1(A_trn,Fr_trn,Fc_trn,latent_dim, Parameters, 500);

% row view
Fr_tst =  U_row(ID_TestRow,:)*sqrt(S_row);
Fr_tst_argu= [ones(size(Fr_tst,1),1)  ,  Fr_tst ];
if size(Fr_tst_argu,2) ~= size(Br,1)
    warning('inner dim');
end
Y_predicted_row = Fr_tst_argu *Br;%

% column view
Fc_tst =  U_col(ID_TestCol,:)*sqrt(S_col);
Fc_tst_argu= [ones(size(Fc_tst,1),1)  ,  Fc_tst ];
if size(Fc_tst_argu,2) ~= size(Bc,1)
    warning('inner dim');
end
Y_predicted_col = Fc_tst_argu *Bc;%

% %
% [U_adj,S_adj,V_adj]= svd(TrnLabelMat_Cleaned);
% [U_adj,S_adj,V_adj,~]=CleanSVD(U_adj,S_adj,V_adj);
% 
% %row view
% Y_trn_row=U_adj*sqrt(S_adj); %trn
% X_trn_row = U_row(ID_TrainRow(idx_Remaining_Row),:)*sqrt(S_row);
% X_tst_row = U_row(ID_TestRow,:)*sqrt(S_row);
% Y_predicted_row = DoPLS(Y_trn_row, X_trn_row, X_tst_row, nComp);
% 
% %column view
% Y_trn_col = V_adj*sqrt(S_adj);
% X_trn_col = U_col(ID_TrainCol(idx_Remaining_Col),:)*sqrt(S_col);
% X_tst_col = U_col(ID_TestCol,:)*sqrt(S_col);
% Y_predicted_col= DoPLS(Y_trn_col,X_trn_col,X_tst_col, nComp);
% 
%
Predicted_LabelMat  =Y_predicted_row *  Y_predicted_col' ; %( sqrt(S_adj)* V_adj'  );  since Y_test already contains sqrt(S_adj)