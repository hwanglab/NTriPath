# NTriPath
"An integrative somatic mutation analysis to identify pathways linked with survival outcomes across 19 cancer types", Bioinformatics, 2015. 

NTriPath is an extension of nonnegative matrix tri-factorization that is designed to identify altered pathways from mutation data. 
It can handle the sparsity of the somatic mutation matrix and the incompleteness of current pathway database annotation 
by incorporating the prior knowledge from human gene–gene interaction networks. For more information, please refer to our article  (https://doi.org/10.1093/bioinformatics/btv692).
 
## Requirements
- To start NtriPath, run main.m (Matlab or Octave, but we have tested our code only on Matlab) 

- TCGA mutation data (X: 4790 patients × 25168 genes): ```mutation_matrices.mat``` (collected from the TCGA data portal on May 19, 2013)
- Pathway datasets (V_0)
 1.  4620 conserved subnetworks: ```bipartite_PPI_module.mat``` 
 2. 186 KEGG pathway dataset version 3: ```bipartite_c2_kegg_v3_broad_CNA.mat``` and ```c2_kegg_v3_curated.mat```
 3.  217 Biocarta pathway dataset version 3:  ```bipartite_c2_biocarta_v3_broad_CNA.mat``` and ```c2_biocarta_v3_curated.mat```
 4. 430 Reactome pathway dataset version 3: ```bipartite_c2_reactome_v3_broad_CNA.mat``` and ```c2_reactome_v3_curated.mat```
 1 is from the human gene–gene interaction network (Suthram etal., 2010) and 2-4 are are from MsigDB (Subramanian etal., 2005). 
- Gene–gene interaction (A: 12456 genes × 12456 genes): ```ppiMatrixTF.mat``` (Zhang et al., 2011, Keshava Prasad et al., 2009,  Rossin et al. 2011)
- Gene name: ```gene_name_info.mat```