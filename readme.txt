% The following demo codes were written by MATLAB. The users can apply them to run four different scenarios of predicting DTIs.
% DTI is the m X n matrix accounting for DTIs between m drugs and n targets
% d_s is the similarity matrix between m drugs
% t_s is the similarity matrix between n targets
% Four benchmark datasets are EN, IC, GPCR, NR.
% Four scenarios are S1, S2, S3 and S4
% nComp is the tunable parameters for our TMF. Its best value depends on the size and the complexity of datasets.
% The default values of nComp for the datasets are 50, 18, 18, 8 respectively.(in additon 4 for S1 only in NR since it is too small) 
% We just tuned it roughly, but the users may tune it to achieve better results.
% The codes follows GNU General Public License.
% Contact: jianyushi@nwpu.edu.cn


load('..\NR.mat')
nComp = 8;
Task = 'S4'; 
CV = 10; %

Scenario = 'S2';
nComp =6;
nCV = 10;   
 tic
    [~, ave_AUC, ave_AUPR] =...
        PredictBySingleBipartite(DTI, (d_s+d_s')/2,(t_s+t_s')/2,Scenario,nCV ,nComp,[],[],false);
    toc
