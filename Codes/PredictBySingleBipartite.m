function [OutputScores, ave_AUC, ave_AUPR,std_AUC, std_AUPR] =PredictBySingleBipartite...
    (Adj, Sim_row,Sim_col,...
    Scenario, nCV, ratio_,...
    RowNames_cell, ColNames_cell,Debug)
% Adj: binary Adjacent matrix (0,1) between two differet nodes, such as drugs and targets
%S_row,S_col: similarity matrices between drugs and between targets.
% nCV  =10; nComp=6;
%  % ratio_ (0,1] :  the coefficient in denominator so as to latent dimentsion = rank(TrnAdj)*nComp
% TODO: refactor the codes to the form of feature
% DEMO: PredictBySingleBipartite(Ard_binary, Sr,Sd,'S2',5 ,2,1,[],[],true);
%
if nargin <9
    Debug = true;
end
if nargin < 8
    ColNames_cell=[];
end
if nargin <7
    RowNames_cell=[];
end


Sim_row =(Sim_row + Sim_row')/2;
Sim_col =(Sim_col + Sim_col')/2;

fixRandom = Debug;
if fixRandom
    disp('Debug')
    rand('state',1234567890); % fix random seed for debug
end

%%%-NOTE- %%% threshold_for_real_valued = 1; % may look like 1.38 to keep continues has
%%%-NOTE- %%% similar number of entries as those of binary

% the first five scenarios are used for cross-validation only.
% the last one is used for finding novel associations
OutputScores = [];
switch upper(Scenario)  %%NOTE: <Easy generated bias in S1>
    % about parameters, % the bigger, the faster
    % use the smaller lamda_r when handling smaller datasets
    case 'S1'
        Parameters.lamda_ar = 1;         Parameters.lamda_ac = 1;
        Parameters.lamda_r =.5;           Parameters.lamda_c = .5; % the bigger, the faster
%         a= [0.005,0.05,0.5,1];
%         b= [0.005,0.05,0.5,1];
%         Parameters.lamda_ar = a(4);         Parameters.lamda_ac = a(4);
%         Parameters.lamda_r =b(4);           Parameters.lamda_c = b(4); 

        [OutputScores, ave_AUC, ave_AUPR,std_AUC, std_AUPR] =PerformS1(Adj, Sim_row,Sim_col, nCV,ratio_,Parameters);
        
    case 'S2'
        % nComp :  the coefficient in denominator as  rank(TrnAdj)/nComp
        Parameters.lamda_ar = 0.05;         Parameters.lamda_ac = 0.05;
        Parameters.lamda_r = 0.5;           Parameters.lamda_c = 0.5;
        [OutputScores, ave_AUC, ave_AUPR,std_AUC, std_AUPR] = PerformS2(Adj, Sim_row,Sim_col, nCV,ratio_,Parameters);
        
    case 'S3'
        Parameters.lamda_ar = 0.5;          Parameters.lamda_ac = 0.5;
        Parameters.lamda_r = 0.05;          Parameters.lamda_c = 0.05;
        [OutputScores, ave_AUC, ave_AUPR,std_AUC, std_AUPR]  =  PerformS2(Adj',Sim_col, Sim_row, nCV,ratio_,Parameters) ;
        OutputScores = transpose(OutputScores);
        
    case 'S4'
        Parameters.lamda_ar = 0.05;         Parameters.lamda_ac = 0.05;
        Parameters.lamda_r = 0.5;           Parameters.lamda_c = 0.5;
        [OutputScores, ave_AUC, ave_AUPR]  = PerformS4(Adj, Sim_row,Sim_col, nCV, ratio_,Parameters);
        
    otherwise
        warning('Scenario is out of range')
end


%% : S1
function [OutputScores,ave_AUC, ave_AUPR,std_AUC, std_AUPR] = PerformS1(Adj, Sim_row,Sim_col, nCV,ratio_,Parameters)
if ratio_ >1
    latent_dim = ratio_; % directly use ratio_ as latent dim
    disp('directly use ratio_ as the latent dim')
end

% 1 - find positive and negative entries 
ZERO_ = 0;
idx_edge = find(Adj > ZERO_);
num_edges =length(idx_edge);
CV_Idx_Edge= GenerateIdxForCV(num_edges,nCV);

idx_nonedge = find(Adj <= ZERO_); % specific negative edges, just like S2
CV_Idx_NonEdge= GenerateIdxForCV(numel(Adj)-num_edges,nCV);

% 2 - turn similarity matrices into featrue matrices
[U_r,S_r,~]=TurnSimBySVD(Sim_row);
[U_c,S_c,~]=TurnSimBySVD(Sim_col);

OutputScores = zeros(size(Adj));
MARK_POS = +9;
MARK_NEG = -9;
% -- run CV
for k_fold = 1:nCV
    % 3.1 - split.
    clear CV_Temp;
    ID_Tst =  idx_edge( CV_Idx_Edge{k_fold} );
    ID_Tst_neg = idx_nonedge( CV_Idx_NonEdge{k_fold} );
    
    %3.2 - training entries are obeserved, and testing entries are unobserved
    % marks of obseved and unobserved = 1, 0 repspectively
    Observed_Flag = ones(size(Adj) ); %contains the combination of S2, S3, S4 block
    Observed_Flag(ID_Tst) = 0;
    Observed_Flag(ID_Tst_neg) = 0;
    
    %3.3 - find pure S1 block in TrnMat, so as to remove S2, S3, S4 in it
    TempMat =  Adj;
    TempMat (ID_Tst)  = ZERO_;% reflect the matrix removed testing edges
    %-% use pure S1
    [trn_row_s1,trn_col_s1] = SplitOutPureS1Scenarios(TempMat); 
    if ratio_<=1 % using the low-rank
        latent_dim=fix(rank(TempMat)* ratio_);
    end
    clear TempMat;
    fprintf('latent dimension r= %d\n',latent_dim)
    %% NOTE::    
    % S1, using observed entries as the training entries and unobserved
    % entries as the testing entries
    
    % 4.1 - set the values to unobserved entries , so as to calcluate the
    % initial values when solving the optimization 
    LabelMat= Adj; % may contain all-zero columns, which cause S4 prediction
    Prior = GetPrior(Adj); %% NOTE %% Prior should not be 0 or 1
    LabelMat(ID_Tst)  = Prior;      LabelMat(ID_Tst_neg)  = Prior;
    
    % 4.2 - run optimization    
    Predicted_LabelMat =PredictS1_Optim...
        ( LabelMat (trn_row_s1,trn_col_s1),Observed_Flag (trn_row_s1,trn_col_s1) ,...
        U_r,S_r, U_c,S_c, ...
        trn_row_s1, trn_col_s1,trn_row_s1, trn_col_s1, latent_dim,Parameters);
    
    % 5 evaluation for only pure S1
    MARK_mat = zeros(size(Adj) ); 
    MARK_mat(ID_Tst) = MARK_POS;
    MARK_mat(ID_Tst_neg) = MARK_NEG;

    idx_pos = find(MARK_mat(trn_row_s1,trn_col_s1) ==MARK_POS);
    idx_neg = find(MARK_mat(trn_row_s1,trn_col_s1) ==MARK_NEG);
    TrueScore = Predicted_LabelMat(idx_pos);
    FalseScore = Predicted_LabelMat(idx_neg);
    
    [AUC_(k_fold), AUPR_(k_fold) ]=EstimationAUC(TrueScore,FalseScore,2000,0);
    
   
    % only output the pure S1 scores
    OutputScores(trn_row_s1,trn_col_s1)  = Predicted_LabelMat;
    fprintf(1,"CV=%d\n",k_fold);
end % CV ends

ave_AUC =mean(AUC_) ; ave_AUPR = mean(AUPR_);
std_AUC = std(AUC_) ; std_AUPR = std(AUPR_);
disp([ave_AUC, ave_AUPR ]);
disp([std_AUC, std_AUPR ]);

%-----------------------------------------------------------------------------------%
function [trn_row_s1,trn_col_s1] = SplitOutPureS1Scenarios(AdjMat)
Col_flag = (sum(AdjMat,1) > 0);
Row_flag = ( sum(AdjMat,2) > 0);

Col_flag_mat = repmat(Col_flag, size(AdjMat,1),1);
Row_flag_mat = repmat(Row_flag, 1, size(AdjMat,2));

% 0 in Flag is the case of S4, while other entries==1 in row are the
% cases of S2, and other entries==1 in col are the cases of S3. The
% entries ==2 are the cases of S1;
ScenarioFlag = Col_flag_mat + Row_flag_mat;

S1_Flag= 2;

% 1: Pure S1
[r_s1,c_s1]=find(ScenarioFlag==S1_Flag);
trn_row_s1 = unique(r_s1);
trn_col_s1 = unique(c_s1);
%
function Prior = GetPrior(Adj)
DiscreteList = unique(Adj(:));
count_ = zeros(length(DiscreteList),1);
for n=1: length(DiscreteList)
    count_(n) =length( find(Adj(:) == DiscreteList(n) ) ) ;
end
Prior = sum(count_ .* DiscreteList)/ sum(count_);
tol_ = 1e-5;
if Prior< tol_
    Prior = tol_;
end

%% : S2/S3
function [OutputScores,ave_AUC, ave_AUPR,std_AUC, std_AUPR] = PerformS2(Adj, Sim_row,Sim_col, nCV,ratio_,Parameters)
[nRow, ~]=size(Adj);
CV_Idx_row= GenerateIdxForCV(nRow,nCV);
OutputScores = zeros(size(Adj));
if ratio_ >1
    latent_dim = ratio_; % directly use ratio_ as latent dim
    disp('directly use ratio_ as the latent dim')
end

% turn the similarity matrix into feature matrix, if using the feature
% matrix, we can comment the codes, and use it as U_row and identity matrix
% as S_row: U_row = F_r; S_row =eye(size(F_r,2));
[U_row,S_row,~]=TurnSimBySVD(Sim_row);
[U_col,S_col,~]=TurnSimBySVD(Sim_col);

for k_fold = 1:nCV
    fprintf('k=%d\n',k_fold)
    % split.
    CV_Temp= CV_Idx_row;    CV_Temp(k_fold) = [];
    ID_Trn_r =  cell2mat(CV_Temp) ;
    clear CV_Temp;
    ID_Tst_r =  CV_Idx_row{k_fold} ;
    
    TrnLabelMat= Adj(ID_Trn_r,:); % may contain all-zero columns, which cause S4 prediction
    TstLabelMat = Adj(ID_Tst_r,:);
    
    % core function
    if ratio_<=1 % using the low-rank
        latent_dim=fix(rank(TrnLabelMat)* ratio_);
    end
    %% run prediction
    Predicted_LabelMat =PredictS2_Optim(TrnLabelMat,U_row,S_row,ID_Trn_r, ID_Tst_r, U_col,S_col,latent_dim,Parameters);
    
    %Measure 2: the evaluation by removing S4 case, so as to obtain pure S2.
    [AUC_(k_fold), AUPR_(k_fold) ] = Measure_S2(Predicted_LabelMat,TrnLabelMat,TstLabelMat);
    % %     disp([AUC_(k_fold), AUPR_(k_fold) ]
    OutputScores(ID_Tst_r,:) = Predicted_LabelMat;
    
end % CV ends

ave_AUC =mean(AUC_) ; ave_AUPR = mean(AUPR_);
std_AUC = std(AUC_) ; std_AUPR = std(AUPR_);
disp([ave_AUC, ave_AUPR ]);
disp([std_AUC, std_AUPR ]);

%% : S4
function [OutputScores,ave_AUC, ave_AUPR] =PerformS4(Adj, Sim_row,Sim_col, nCV,ratio_,Parameters)
% nComp = 5; % parameter for regression
% Codes have been validated by comparing with the original DrugRepositioningForS4
if ratio_ >1
    latent_dim = ratio_; % directly use ratio_ as latent dim
    disp('directly use ratio_ as the latent dim')
end

[nRow, nCol]=size(Adj);
CV_Idx_row= GenerateIdxForCV(nRow,nCV);
CV_Idx_col= GenerateIdxForCV(nCol,nCV);
OutputScores = zeros(size(Adj));

[U_r,S_r,~]=TurnSimBySVD(Sim_row);
[U_c,S_c,~]=TurnSimBySVD(Sim_col);

NO_POS_IN_TEST = -1;

for k_fold_r = 1:nCV
    
    % find training entries of both interactions and non-interactions.
    % (1) firstly, find the training drugs and the testing drug to perform
    % S2 test.
    CV_Temp= CV_Idx_row;    CV_Temp(k_fold_r) = [];
    ID_Trn_r =  cell2mat(CV_Temp) ;     clear CV_Temp;
    ID_Tst_r =  CV_Idx_row{k_fold_r} ;
    
    % (2) assign the interactions between the testing targets in S4 and the
    % training drugs with ZEROS
    for k_fold_c = 1:nCV
        CV_Temp= CV_Idx_col;    CV_Temp(k_fold_c) = [];
        ID_Trn_c =  cell2mat(CV_Temp) ; clear CV_Temp;
        ID_Tst_c =  CV_Idx_col{k_fold_c} ;
        
        
        TrnLabelMat= Adj(ID_Trn_r,ID_Trn_c); % may contain all-zero columns, which cause S4 prediction
        TstLabelMat = Adj(ID_Tst_r,ID_Tst_c); % S4 block
        
        if ratio_<=1 % using the low-rank
            latent_dim=fix(rank(TrnLabelMat)* ratio_);
        end
        Predicted_LabelMat = PredictS4_Optim(...
            TrnLabelMat,U_r,S_r, U_c,S_c, ID_Trn_r, ID_Trn_c,ID_Tst_r, ID_Tst_c,latent_dim,Parameters);
        
        % MeasureS4
        [AUC_(k_fold_r,k_fold_c), AUPR_(k_fold_r,k_fold_c)] = Measure_S4(Predicted_LabelMat,TstLabelMat,NO_POS_IN_TEST) ;
        OutputScores(ID_Tst_r,ID_Tst_c)     = Predicted_LabelMat;
    end
    
end

ave_AUC =mean(AUC_(AUPR_~= -1) ) ;
ave_AUPR = mean(AUPR_( AUPR_~= -1 ));

disp([ave_AUC, ave_AUPR ]);
disp([std(AUC_(:)), std(AUPR_(:)) ]);


%% : helper
function [U,S,V]=TurnSimBySVD(Sim)
[U,S,V]= svd(Sim);
[U,S,V,~]=CleanSVD(U,S,V);

%%
function [AUC_S2,AUPR_S2] = Measure_S2(Predicted_Scores,trn_Adj,tst_Adj)
%% global measure
threshold  =1;
degrees_= sum(trn_Adj,1); % to remove S4 cases, if occured.
Scores  = Predicted_Scores(:,degrees_>=threshold);
Label_mat = tst_Adj(:,degrees_>=threshold);

TrueScore = Scores( Label_mat(:)>0);
FalseScore= Scores( Label_mat(:)==0);

[AUC_S2, AUPR_S2 ]=EstimationAUC(TrueScore,FalseScore,2000,0);

%%
function [AUC_S4,AUPR_S4] = Measure_S4(Predicted_Scores,tst_Adj,NO_POS_IN_TEST)
Pos_Num= sum(tst_Adj(:)); % to remove all-zero blocks in S4 , if occured.
if Pos_Num>=1
    TrueScore = Predicted_Scores( tst_Adj(:)>0);
    FalseScore= Predicted_Scores( tst_Adj(:)==0);
    [AUC_S4, AUPR_S4 ]=EstimationAUC(TrueScore,FalseScore,2000,0);
else % cannot measure when no positive sample, using flag.
    AUC_S4= NO_POS_IN_TEST;
    AUPR_S4= NO_POS_IN_TEST;
    warning('cannot measure because of no positive sample in the testing set.')
end
