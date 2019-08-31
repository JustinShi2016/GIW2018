function Predicted_LabelMat = PredictS1_Optim(TrnLabelMat,Observed_Flag,...
    U_row,S_row, U_col,S_col,...
    ID_TrainRow, ID_TrainCol,ID_TestRow, ID_TestCol, threshold, Parameters)
% TrnLabelMat - pure S1 block, has no isolated node.
if nargin < 11 || isempty(threshold)
    threshold=fix(rank(TrnLabelMat)/4);
end
latent_dim =max(1, min( threshold, rank(TrnLabelMat) ) );
% disp('Parameter of PLS for S1')
% disp(nComp)
%row view
F_trn_row = U_row(ID_TrainRow,:)*sqrt(S_row);
F_tst_row = U_row(ID_TestRow,:)*sqrt(S_row);
%column view
F_trn_col = U_col(ID_TrainCol,:)*sqrt(S_col);
F_tst_col = U_col(ID_TestCol,:)*sqrt(S_col);%

[U_adj,S_adj,V_adj]= svd(TrnLabelMat);
[U_adj,S_adj,V_adj,~]=CleanSVD(U_adj,S_adj,V_adj);

A = TrnLabelMat;
Fr = F_trn_row;
Fc = F_trn_col;
[Br_trn,Bc_trn,Ar_trn,Ac_trn] = Solver_optim_v1_entry(A,Observed_Flag,Fr,Fc, latent_dim, Parameters,500);

%% only this part is different to S4
% % % %  Predicted_LabelMat  =Y_trn_row* Y_predicted_col';
% % %   Predicted_LabelMat  =max( (Y_predicted_row * Y_trn_col' )  ,  (Y_trn_row* Y_predicted_col' ) ) ; 
% %   
%    Predicted_LabelMat  = ([ones(size(Fr,1),1)  ,  Fr ] *Br_trn) *  ([ones(size(Fc,1),1)  ,  Fc ] *Bc_trn)'  ; % better

Predicted_LabelMat  =Ar_trn * Ac_trn' ; % best?
%     Predicted_LabelMat  =( ([ones(size(Fr,1),1)  ,  Fr ] *Br_trn) * Ac_trn' ...
%         +  Ar_trn* ([ones(size(Fc,1),1)  ,  Fc ] *Bc_trn)' )/2 ; % better
%    Predicted_LabelMat  = ([ones(size(Fr,1),1)  ,  Fr ] *Br_trn) *  ([ones(size(Fc,1),1)  ,  Fc ] *Bc_trn)'  ; % better

%   

  