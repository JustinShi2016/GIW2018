% Comparing

% 
clear Repeat Scenario nComp nCV ave_AUC ave_AUPR
Repeat =50;
Scenario = 'S2';
nComp =6;
nCV = 10;

for k=1:Repeat
     fprintf(1,'K=%d \n',k);
     tic
    [~, ave_AUC(k), ave_AUPR(k)] =...
        PredictBySingleBipartite(DTI, (d_s+d_s')/2,(t_s+t_s')/2,Scenario,nCV ,nComp,[],[],[],false);
    toc
end

disp('===================================')
fprintf(1,'%s, %d-CV, %d Repeats \n',Scenario,nCV, Repeat);
disp([mean(ave_AUC), mean(ave_AUPR)])
disp([std(ave_AUC), std(ave_AUPR)])
% disp( ave_AUC)
% disp(ave_AUPR)