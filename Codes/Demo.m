Scenario = 'S2';
nComp =6;
nCV = 10;   
   tic
    [~, ave_AUC, ave_AUPR] =...
        PredictBySingleBipartite(DTI, (d_s+d_s')/2,(t_s+t_s')/2,Scenario,nCV ,nComp,[],[],[],false);
    toc
