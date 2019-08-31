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
