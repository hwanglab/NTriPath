clc; close all; clear all;

% Network regularized non-negative TRI matrix factorization for PATHway
% identification (NTriPath)
%
% Written by Sunho Park (sunho.park@utsouthwestern.edu)
%            Clinical Sciences department,
%            University of Texas Southwestern Medical Center
%
% This software implements the network regularized non-negative tri-matrix factorization method 
% which integrates somatic mutation data with human protein-protein interaction networks and a pathways database. 
% Driver pathways altered by somatic mutations across different types of cancer
% can be identified by solving the following problem:
%  
%  \min_{S>=0,V>=0} ||X - USV'||_W^2 
%                   + lambda_S ||S||_1^2 + lambda_V||V||_1^2
%                   + lambda_{V0}||V - V0||_F^2 + \lambda_{V_L}tr(V'LV)
%
% where ||X - USV'||_W^2 = sum_i sum_j W_ij(X_ij - [USV]_ij)^2   
%
% Reference:
%      "An integrative somatic mutation analysis to identify pathways 
%          linked with survival outcomes across 19 cancer types"
%      submitted to Nature Communication 
%
% Please send bug reports or questions to Sunho Park.
% This code comes with no guarantee or warranty of any kind.
%
% Last modified 11/20/2014

 
%--- general setting()
% we consider different pathway databases 
m_cPriorNames = {'ppi_module', 'kegg', 'reactome', 'biorcrata'};
m_str_prior = m_cPriorNames{1};

%--- load mutation data
% X: Somatic mutation data (#patients X #genes)
% U0: Patient cluster obtained from patient’s clinical information (#patients X #cancer)

load ./data/mutation_matrices

X = full(m_cdata.X);

U0 = full(m_cdata.F);
U0 = U0./repmat(sum(U0,1),[size(U0,1),1]);

% we consider genes in ***
gene_name = m_cdata.geneID';
load ./data/gene_name_info
[common_gene, ai, bi] = intersect(gene_name, gene_name_chr(:,1));
X = X(:,ai);

%--- load a pathway database 
disp(['the selected V0: ' m_str_prior]); 

switch m_str_prior,
    % ppi network (# pathway: 4620)
    case 'ppi_module'
        load ./data/bipartite_PPI_module;
                
    % kegg (# pathway: 186)
    case 'kegg', 
        load ./data/bipartite_c2_kegg_v3_broad_CNA
        load ./data/c2_kegg_v3_curated.mat
        
    % reactome (# pathway: 430)
    case 'reactome', 
        load ./data/bipartite_c2_reactome_v3_broad_CNA
        load ./data/c2_reactome_v3_curated
        
    % biorcrata (# pathway: 217)        
    case 'biorcrata'
        load ./data/bipartite_c2_biocarta_v3_broad_CNA
        load ./data/c2_biocarta_v3_curated
         
    % default    
    otherwise,
        load ./data/bipartite_PPI_module;   
end

module_idx = module_idx(:,bi);

%--- Generate Laplican matrix (ppi network) 8/20/2013
% A: Gene-gene interaction network (#genes X #genes types)
% Laplaican matrix L: L = diag(sum(A,2)) - A

load ./data/ppiMatrixTF % ppi network
[ci, ai, bi] = intersect(common_gene, gene_name_ge);
m_cSelected_gene = ci;
   
A = sparse(ppiMatrixTF(bi,bi));
D = sum(A, 2);
 
% Somatic mutation data (#patients X #genes) 
X = double(X(:,ai));
X = sparse(X/sum(sum(X)));

% V0 : Initial pathway information (#pathways X #genes)
V0 = module_idx(:,ai)';
V0 = V0/sum(sum(V0));
V0 = full(V0);

[N, K1] = size(U0);
[M, K2] = size(V0);

L = spdiags(D,0,M,M) - A;

%--- Nonnegative Matrix Tri-Factorization (NMTF)
% regularization parameters
lamda_V = 1;
lamda_V0 = 0.1; 
lamda_VL = 1;

lamda_S = 1;

% the maximum iteration for NtriPath  
Maxiters = 20;
     
fprintf('lamda_V = %3.2f, lamda_{V0} = %3.2f, lamda_{VL} = %3.2f, lamda_S = %3.2f \n',...
         lamda_V, lamda_V0, lamda_VL, lamda_S);
disp(['[iter] obj = (||X-USV||_W^2, ',...
      '||S||_1^2, ||V||_1^2, ||V-V_0||_F^2, tr(V^tLV))']);
Obj = zeros(5,1);
                        
%--- initilaize parameters in the procedure for the inadmissible zeros problem
% kappa: Inadmissible structural zero avoidance adjustment (e.g. 1e-6)
% kappa_tol: Tolerance for identifying a potential structural nonzero (e.g., 1e-10)
% eps: minimum divisor to prevent divide-by-zero (e.g. 1e-10)
kappa = 1e-6;       
kappa_tol = 1e-10;  
eps = 1e-10;

%--- initilaize factor matrices
% U: Patient cluster (#patients X #cancer)
% S: Cancer type-pathway association matrix (#cancer types X #pathways)
% V: Updated pathway information (#pathways X #genes)
% Here, we assume that U is fixed during the learning process. 
U = max(U0, kappa);
S = ones(K1, K2);                     
V = max(V0, kappa);                        

% W (weight matrix): W_ij is 1 if X_ij is non zero, otherwise 0                    
W = X > 0;
W_zero = W == 0;
                                              
for iter = 1:Maxiters,
    %--- update S
    % X_hat = (U*S*V')oW, where o is an element wise multiplication operator 
    X_hat = (U*S)*V';
    X_hat(W_zero) = 0;
                            
    % multiplicative factor
    gamma_S = ((U'*X)*V)./( ((U'*X_hat)*V) + lamda_S*sum(sum(S)) + eps);
                            
    % checking inadmissible zeros
    ChkIDX = (S<kappa_tol) & (gamma_S>1);
                            
    S(ChkIDX) = (S(ChkIDX)+kappa).*gamma_S(ChkIDX);
    S(~ChkIDX) = S(~ChkIDX).*gamma_S(~ChkIDX);   
                           
    %- update V                  
    US = U*S;
                            
    X_hat = US*V';
    X_hat(W_zero) = 0;
              
    % multiplicative factor                            
    gamma_V = ( (X'*US) + lamda_V0*V0 + lamda_VL*(A*V) )./...
                        ( (X_hat'*US) + lamda_V0*V ...
                                      + lamda_VL*(spdiags(D,0,M,M)*V) + lamda_V*sum(sum(V)) + eps );        
    % checking inadmissible zeros
    ChkIDX = (V<kappa_tol) & (gamma_V>1);
                            
    V(ChkIDX) = (V(ChkIDX)+kappa).*gamma_V(ChkIDX);
    V(~ChkIDX) = V(~ChkIDX).*gamma_V(~ChkIDX);
    
    % normalization of V and S
    V_sum = sum(V, 1);
    V = V./repmat(V_sum,[M,1]);
    S = S.*repmat(V_sum,[K1,1]);

    %- diplay the objective function value
    if iter > 0	
       X_hat = (U*S)*V';
                            
       Obj(1) = sum( (X(W) - X_hat(W)).^2 ); % ||X-(U*S)*V'||_W^2 
       Obj(2) = lamda_S*( sum(sum(S))^2 );
       Obj(3) = lamda_V*( sum(sum(V))^2 );
       Obj(4) = lamda_V0*sum(sum( (V-V0).^2 ));
       Obj(5) = lamda_VL*( sum(sum(V.*(L*V))) );                           
                            
       str = '[%d iter] obj: %3.3f = (%3.9f, %3.9f, %3.9f, %3.9f, %3.9f) \n';
       fprintf(str, iter, sum(Obj), Obj(:));
    end
end
                    
disp('%-----------------------------------------------------------------%')
%--- update pathway information from the factor matrix (solution) V 
Threshold = 1e-2;                       
Vadded = (V > V0 + Threshold) & (V0 == 0);
fprintf('The number of the newly added genes: %d \n', sum(sum(Vadded)));

%--- display the results 
% the results for each cancer type are displayed as followed.
% ex. >> A: full cancer type name (#: b)
% ex. >> rank: cth, D (e+f/g): H(*)-i,
% explanations:
% A: the cancer type
% b: the number of patients with the cancer type A 
% c: the rank of the current pathway
% D: the name of the current pathway (if available)
% e: the number of genes in the current pathway (after filtering, we only consider genes in both the ppi network and the pathway database)
% f: the number of newly added genes into the current pathway
% g: the number of genes in the current pathway (in the orginal pathway database)
% H: the gene symbol in the current pathway
% i: the percentage of patients who have a mutation at the gene H
% *: if the gene H is an added one, the gene symbol is displayed with a star

for m_ni = 1:K1,
    % patients with the ith cancer type 
    m_vIDX_cl = U0(:,m_ni) > 0;
    m_nNumpat = sum(m_vIDX_cl);
                            
    m_strname = m_cdata.className{m_ni};
    disp(['>> ', m_strname, '(#: ' num2str(m_nNumpat), ')']);
    
    % Abbreviated cancer type 
    m_strNameTok = strtok(m_strname,':');
    
    [val, idx] = sort(S(m_ni,:), 'descend');
    
    for m_nsub = 1:20,
        m_nSelIDX = idx(m_nsub);
        
        if ~strcmpi(m_str_prior,'ppi_module'), % only for when the pathways are known
            m_cgeneSet = pathway_gname{m_nSelIDX};
            
            [m_cIntsect, ai, bi] = intersect(m_cSelected_gene, m_cgeneSet);
            m_nlen_genes_or = length(m_cgeneSet);
            m_nlen_genes_co = length(m_cIntsect);
            
            m_str_pathwayname = pathway_name{m_nSelIDX};
        else
            m_str_pathwayname = ['PPI-', num2str(m_nSelIDX), 'th-module'];
            
            m_cIntsect = m_cSelected_gene(V0(:,m_nSelIDX)>0);
            
            m_nlen_genes_or = length(m_cIntsect);
            m_nlen_genes_co = length(m_cIntsect);
        end
                                
      % added
      m_vnewIDX = find(Vadded(:,m_nSelIDX));
                                
      m_nlen_genes_new = m_nlen_genes_co;
      if ~isempty(m_vnewIDX)
         m_nlen_genes_new = m_nlen_genes_co + length(m_vnewIDX);
                                    
         for m_ninner = 1:length(m_vnewIDX),
             m_cIntsect{m_nlen_genes_co+m_ninner} ...
                    = [m_cSelected_gene{m_vnewIDX(m_ninner)}, '*'];
         end
      end
         
      m_vCntGenes = zeros(m_nlen_genes_new,1);
      for m_ninner = 1:m_nlen_genes_new
          if m_ninner > m_nlen_genes_co,
              m_vIDX_ge = m_vnewIDX(m_ninner-m_nlen_genes_co);
          else
              m_vIDX_ge = ismember(m_cSelected_gene, m_cIntsect{m_ninner});
          end
          
          m_vCntGenes(m_ninner) = sum(X(m_vIDX_cl, m_vIDX_ge)>0);
      end
      
      m_str_discribe = 'rank: %dth, %s (%d+%d/%d): ';
      m_str_subs = repmat('%s-%s, ', [1, m_nlen_genes_new]);
                                m_str_subs(end-1:end) = '\n';
                                
      m_cdisplay = cell(2*m_nlen_genes_new,1);
      for m_ninner = 1:m_nlen_genes_new
 	  m_cdisplay{(m_ninner-1)*2 + 1} = m_cIntsect{m_ninner};
          m_cdisplay{(m_ninner-1)*2 + 2} = num2str(100*(m_vCntGenes(m_ninner)/m_nNumpat));
      end
                                
      m_strdisp = sprintf([m_str_discribe, m_str_subs] , m_nsub, m_str_pathwayname, ...
                           m_nlen_genes_co, length(m_vnewIDX), m_nlen_genes_or, m_cdisplay{:});
                                                                                                                 
      disp(m_strdisp(1:end-1));
    end
end                         
                         
                        


 

