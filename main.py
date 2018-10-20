import numpy as np
import scipy.io
from scipy import sparse

'''
Network regularized non-negative TRI matrix factorization for PATHway
identification (NTriPath)

Written by Sunho Park (parks@ccf.org),
           Quantitative Health Sciences,
           Cleveland Clinic Lerner Research Institute
           Nabhonil Kar (nkar@princeton.edu),
           Princeton University

This software implements the network regularized non-negative tri-matrix factorization method 
which integrates somatic mutation data with human protein-protein interaction networks and a pathways database. 
Driver pathways altered by somatic mutations across different types of cancer
can be identified by solving the following problem:
  
 \min_{S>=0,V>=0} ||X - USV'||_W^2
                  + lambda_S ||S||_1^2 + lambda_V||V||_1^2
                  + lambda_{V0}||V - V0||_F^2 + \lambda_{V_L}tr(V'LV)

where ||X - USV'||_W^2 = sum_i sum_j W_ij(X_ij - [USV]_ij)^2   

Reference:
    "An integrative somatic mutation analysis to identify pathways 
        linked with survival outcomes across 19 cancer types"
    submitted to Nature Communication 

Please send bug reports or questions to Sunho Park.
This code comes with no guarantee or warranty of any kind.

Last modified 10/19/18
'''

# --- general setting()
# we consider different pathway databases
m_cPriorNames = ['ppi_module', 'kegg', 'reactome', 'biorcrata']
m_str_prior = m_cPriorNames[0]

# --- load mutation data
# X: Somatic mutation data (#patients X #genes)
# U0: Patient cluster obtained from patient�s clinical information (#patients X #cancer)
mutation_matrices = scipy.io.loadmat('./data/mutation_matrices')
m_cdata = mutation_matrices['m_cdata']

X = m_cdata['X'][0, 0]

U0 = m_cdata['F'][0, 0]
U0 = U0.todense()
U0 = U0 / np.tile(sum(U0, 0), [U0.shape[0], 1])

# we consider genes in ***
gene_name = m_cdata['geneID'][0, 0].T
gene_name_info = scipy.io.loadmat('./data/gene_name_info')
gene_name_chr = gene_name_info['gene_name_chr']
common_gene, ai, bi = np.intersect1d(gene_name, gene_name_chr[:, 0], return_indices=True)
X = X[:, ai]

# --- load a pathway database
print('the selected V0: ' + m_str_prior)

# ppi network (# pathway: 4620)
if m_str_prior == 'ppi_module':
    ppi_1 = scipy.io.loadmat('./data/bipartite_PPI_module')

# kegg (# pathway: 186)
elif m_str_prior == 'kegg':
    ppi_1 = scipy.io.loadmat('./data/bipartite_c2_kegg_v3_broad_CNA')
    ppi_2 = scipy.io.loadmat('./data/c2_kegg_v3_curated')

# reactome (# pathway: 430)
elif m_str_prior == 'reactome':
    ppi_1 = scipy.io.loadmat('./data/bipartite_c2_reactome_v3_broad_CNA')
    ppi_2 = scipy.io.loadmat('./data/c2_reactome_v3_curated')

# biorcrata (# pathway: 217)
elif m_str_prior == 'biorcrata':
    ppi_1 = scipy.io.loadmat('./data/bipartite_c2_biocarta_v3_broad_CNA')
    ppi_2 = scipy.io.loadmat('./data/c2_biocarta_v3_curated')

# default
else:
    ppi_1 = scipy.io.loadmat('./data/bipartite_PPI_module')

module_idx = ppi_1['module_idx'][:, bi]

# --- Generate Laplican matrix (ppi network) 8/20/2013
# A: Gene-gene interaction network (#genes X #genes types)
# Laplaican matrix L: L = diag(sum(A,2)) - A

ppiMatrixTF = scipy.io.loadmat('./data/ppiMatrixTF')    # ppi network
ci, ai, bi = scipy.intersect1d(common_gene, ppiMatrixTF['gene_name_ge'], return_indices=True)
m_cSelected_gene = ci

A = sparse.csc_matrix(ppiMatrixTF['ppiMatrixTF'][np.ix_(bi, bi)])
D = A.sum(0)

# Somatic mutation data (#patients X #genes)
X = X[:, ai]
X = X / X.sum()

# V0 : Initial pathway information (#pathways X #genes)
V0 = module_idx[:, ai].T
if m_str_prior == 'ppi_module':
    V0 = V0.toarray()
V0 = V0 / V0.sum()

N, K1 = U0.shape
M, K2 = V0.shape

L = (scipy.sparse.spdiags(D, 0, M, M) - A).toarray()
L = sparse.csr_matrix(np.where(L == 2**32-1, -1, L))

# --- Non-negative Matrix Tri-Factorization (NMTF)
# regularization parameters
lamda_V = 1
lamda_V0 = 0.1
lamda_VL = 1

lamda_S = 1

# the maximum iteration for NtriPath
Maxiters = 20

print('lamda_V = %3.2f, lamda_{V0} = %3.2f, lamda_{VL} = %3.2f, lamda_S = %3.2f \n' %
      (lamda_V, lamda_V0, lamda_VL, lamda_S))
print('[iter] obj = (||X-USV||_W^2, ||S||_1^2, ||V||_1^2, ||V-V_0||_F^2, tr(V^tLV))')
Obj = np.zeros((5, 1))

# --- initialize parameters in the procedure for the inadmissible zeros problem
# kappa: Inadmissible structural zero avoidance adjustment (e.g. 1e-6)
# kappa_tol: Tolerance for identifying a potential structural nonzero (e.g., 1e-10)
# eps: minimum divisor to prevent divide-by-zero (e.g. 1e-10)
kappa = 1e-6
kappa_tol = 1e-10
eps = 1e-10

# --- initialize factor matrices
# U: Patient cluster (#patients X #cancer)
# S: Cancer type-pathway association matrix (#cancer types X #pathways)
# V: Updated pathway information (#pathways X #genes)
# Here, we assume that U is fixed during the learning process.
U = np.maximum(U0, kappa)
S = np.ones((K1, K2))
V = np.maximum(V0, kappa)

# W(weight matrix): W_ij is 1 if X_ij is non zero, otherwise 0
W = X > 0
W = W.tocoo()
W_zero = ~(W.todense())

for itera in range(Maxiters):
    # --- update S
    # X_hat = (U*S*V')oW, where o is an element wise multiplication operator
    X_hat = (U@S)@V.T
    X_hat[W_zero] = 0

    # multiplicative factor
    gamma_S = np.divide(((U.T@X)@V), (((U.T@X_hat)@V) + lamda_S*np.sum(np.sum(S)) + eps)).getA()

    # checking inadmissible zeros
    ChkIDX = (S < kappa_tol) & (gamma_S > 1)
    S[ChkIDX] = (S[ChkIDX]+kappa)*gamma_S[ChkIDX]
    S[~ChkIDX] = S[~ChkIDX]*gamma_S[~ChkIDX]

    # update V
    US = U@S

    X_hat = US@V.T
    X_hat[W_zero] = 0

    # multiplicative factor
    gamma_V = np.divide(((X.T@US) + lamda_V0*V0 + lamda_VL*(A@V)),
                        ((X_hat.T@US) + lamda_V0*V + lamda_VL * (sparse.spdiags(D, 0, M, M).toarray()@V) +
                         lamda_V * np.sum(np.sum(V)) + eps)).getA()

    # checking inadmissible zeros
    ChkIDX = (V < kappa_tol) & (gamma_V > 1)

    V[ChkIDX] = (V[ChkIDX] + kappa) * gamma_V[ChkIDX]
    V[~ChkIDX] = V[~ChkIDX] * gamma_V[~ChkIDX]

    # normalization of V and S
    V_sum = np.sum(V, 0)
    V = V/np.tile(V_sum, (M, 1))
    S = S*np.tile(V_sum, (K1, 1))

    # display the objective function value
    if itera >= 0:
        X_hat = (U @ S) @ V.T
        Obj[0] = np.sum(np.square(X[W.row, W.col] - X_hat[W.row, W.col]))   # | | X - (U * S) * V'||_W^2
        Obj[1] = lamda_S*(np.sum(S)**2)
        Obj[2] = lamda_V*(np.sum(V)**2)
        Obj[3] = lamda_V0*np.sum((V-V0)**2)
        Obj[4] = lamda_VL*(np.sum(np.multiply(V, L@V)))

        string = '[%d iter] obj: %3.3f = (%3.9f, %3.9f, %3.9f, %3.9f, %3.9f)'
        print(string % (itera + 1, np.sum(Obj), Obj[0], Obj[1], Obj[2], Obj[3], Obj[4]))

print('%-----------------------------------------------------------------%')
# --- update pathway information from the factor matrix (solution) V
Threshold = 1e-2
Vadded = (V > V0 + Threshold) & (V0 == 0)
print('The number of the newly added genes: %d \n' % np.sum(np.sum(Vadded)))

# --- display the results
# the results for each cancer type are displayed as followed.
# ex. >> A: full cancer type name (#: b)
# ex. >> rank: cth, D (e+f/g): H(*)-i,
# explanations:
# A: the cancer type
# b: the number of patients with the cancer type A
# c: the rank of the current pathway
# D: the name of the current pathway (if available)
# e: the number of genes in the current pathway (after filtering, we only consider genes in both the ppi network
#    and the pathway database)
# f: the number of newly added genes into the current pathway
# g: the number of genes in the current pathway (in the original pathway database)
# H: the gene symbol in the current pathway
# i: the percentage of patients who have a mutation at the gene H
# *: if the gene H is an added one, the gene symbol is displayed with a star

for m_ni in range(K1):
    # patients with the ith cancer type
    m_vIDX_cl = U0[:, m_ni] > 0
    m_nNumpat = np.sum(m_vIDX_cl)

    m_strname = m_cdata['className'][0, 0][m_ni][0][0]
    print('>> %s (#: %d)' % (m_strname, m_nNumpat))

    # Abbreviated cancer type
    m_strNameTok = m_strname.split(':')[0]

    idx = np.argsort(S[m_ni, :], kind='mergesort')[::-1]
    val = np.sort(S[m_ni, :], kind='mergesort')[::-1]

    for m_nsub in range(20):
        m_nSelIDX = idx[m_nsub]

        if not m_str_prior.lower() == 'ppi_module':     # only for when the pathways are known
            m_cgeneSet = ppi_2['pathway_gname'][0][m_nSelIDX]

            m_cIntsect, ai, bi = scipy.intersect1d(m_cSelected_gene, m_cgeneSet, return_indices=True)
            m_nlen_gene_or = len(m_cgeneSet)
            m_nlen_gene_co = len(m_cIntsect)

            m_str_pathwayname = ppi_2['pathway_name'][0][m_nSelIDX][0]
        else:
            m_str_pathwayname = 'PPI-' + str(m_nSelIDX + 1) + 'th-module'

            m_cIntsect = m_cSelected_gene[V0[:, m_nSelIDX] > 0]

            m_nlen_gene_or = len(m_cIntsect)
            m_nlen_gene_co = len(m_cIntsect)

        # added
        m_vnewIDX = np.flatnonzero(Vadded[:, m_nSelIDX])

        m_nlen_genes_new = m_nlen_gene_co
        if len(m_vnewIDX) > 0:
            m_nlen_genes_new = m_nlen_gene_co + len(m_vnewIDX)

            for m_ninner in range(len(m_vnewIDX)):
                m_cIntsect = np.append(m_cIntsect, [m_cSelected_gene[m_vnewIDX[m_ninner]]])

        m_vCntGenes = np.zeros((m_nlen_genes_new, 1))
        for m_ninner in range(m_nlen_genes_new):
            if m_ninner >= m_nlen_gene_co:
                m_vIDX_ge = m_vnewIDX[m_ninner-m_nlen_gene_co]
            else:
                m_vIDX_ge = np.in1d(m_cSelected_gene, m_cIntsect[m_ninner])
            m_vCntGenes[m_ninner] = (X[m_vIDX_cl.nonzero(), m_vIDX_ge.nonzero() > 0).sum()

        m_str_discribe = 'rank: %dth, %s (%d+%d/%d): '
        m_str_subs = ', '.join(['%s-%s'] * m_nlen_genes_new)

        m_cdisplay = []
        for m_ninner in range(m_nlen_genes_new):
            if m_ninner >= m_nlen_gene_co:
                m_cdisplay.append(str(m_cIntsect[m_ninner])+'*')
            else:
                m_cdisplay.append(str(m_cIntsect[m_ninner][0]))
            m_cdisplay.append(str(float(100 * (m_vCntGenes[m_ninner]/m_nNumpat))))

        print(m_str_discribe % (m_nsub + 1, m_str_pathwayname, m_nlen_gene_co, len(m_vnewIDX), m_nlen_gene_or), end="")
        print(m_str_subs % tuple(m_cdisplay))
