# RANKOR: Direct Drug Prioritization from Bulk and Single-Cell Transcriptomic Signatures
 
![RANKOR](RANKOR_OVERVIEW.png)


## Overview

RANKOR is a machine learning framework for **direct drug prioritization from bulk and single-cell transcriptomic signatures**.  
Unlike traditional enrichment or signature-matching approaches, RANKOR learns structured representations of gene expression and chemical space to efficiently rank candidate compounds.

The framework constructs two aligned latent spaces:
- a **transcriptomic latent space** capturing biological mechanisms of action  
- a **chemical latent space** representing drug structures  

A cross-modal mapping connects both spaces, allowing transcriptomic signatures to be projected into chemical space, where drugs are ranked based on similarity.

RANKOR achieves **performance comparable to GSEA**, while offering:
- strong generalization to unseen compounds and cellular contexts  
- **orders-of-magnitude faster runtime**  
- biologically meaningful gene-level interpretability  

Overall, RANKOR enables scalable and flexible **transcriptomics-driven drug discovery and repurposing**.

---

## Key Contributions

- **Direct drug prioritization**  
  Formulates drug ranking as the primary task, avoiding indirect inference via enrichment or similarity matching.

- **Cross-modal representation learning**  
  Aligns transcriptomic and chemical latent spaces to connect gene expression signatures with candidate drugs.

- **Generalization to unseen compounds**  
  Enables prioritization of drugs without prior transcriptomic profiling using chemical structure embeddings.

- **Scalability and efficiency**  
  Achieves orders-of-magnitude faster inference compared to enrichment-based methods such as GSEA.

- **Robustness to noise**  
  Maintains stable performance under moderate perturbations in gene expression data.

- **Single-cell applicability**  
  Supports drug prioritization at the level of patient-specific cell states and subpopulations.

- **Interpretability**  
  Provides gene-level attribution analyses highlighting biologically meaningful transcriptional programs.
