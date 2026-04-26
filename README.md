# RANKOR: Direct Drug Prioritization from Bulk and Single-Cell Transcriptomic Signatures
 
<p align="center">
  <img src="RANKOR_OVERVIEW.png" width="500"/>
</p>



<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>


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


## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate rankor
pip install -r requirements.txt
```

---

## Usage

### Quick Start (Inference Only)

#### 1. Prepare input gene profile

Provide a CSV file with two columns:

```csv
gene,value
STAT1,1.23
IFIT3,0.87
...
```

Missing genes are automatically filled with 0.

---

#### 2. Provide reference dataset

A reference `.h5ad` file is required to define:
- gene ordering
- drug library

Example:

```
splits/train_TAS_high_fold1.h5ad
```

---

#### 3. Use pretrained models

RANKOR requires:

- `checkpoints/expression_encoder.pth`  
- `checkpoints/drug_encoder.pth`  
- `checkpoints/drug_prioritisation.pth`  

---

## How RANKOR Works

1. Load reference dataset  
2. Align input gene profile  
3. Encode transcriptomic signature  
4. Encode drugs from SMILES  
5. Map transcriptomic → chemical space  
6. Rank drugs via cosine similarity  

---
### One-command inference with `RANKOR_prior.py`

For convenience, RANKOR also provides a single-file entry point:

```bash
python RANKOR_prior.py \
  --input path/to/profile.csv \
  --train_h5ad train_fold.h5ad \
  --expression_encoder_ckpt checkpoints/expression_encoder.pth \
  --drug_encoder_ckpt checkpoints/drug_encoder.pth \
  --regressor_ckpt checkpoints/drug_prioritisation.pth \
  --out_csv out_ranked_drugs.csv \
  --topk 100 \
  --print_top 20
```

This command performs the full inference workflow: it loads the reference gene set and drug library, aligns the input gene profile, embeds the transcriptomic signature, maps it into chemical space, and returns a ranked list of candidate drugs.

The output file contains:

- `rank`
- `drug_name`
- `cosine_similarity`

To see all available options:

```bash
python RANKOR_prior.py --help
```

---


## Full Pipeline (Reproducibility)



## Notes
- `--train_h5ad` is used to define the reference **gene list/order** and the **drug library** (`cmap_name`, `canonical_smiles`).
- Fingerprints are **Morgan radius=1, nBits=256**.

## Training (optional)
If you want to reproduce training, run scripts in this order:

1. `scripts/01_preprocess_lincs.py`
2. `scripts/02_prepare_tas_high_splits.py`
3. `scripts/03_train_transcriptome_latent_space.py`
4. `scripts/05_train_chemical_latent_space.py`
5. `scripts/06_train_cross_space_mapping.py`
6. `scripts/07_test_drug_prioritisation.py`



### Step 1 — Download LINCS L1000 data

```bash
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_cp_n720216x12328.gctx
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/siginfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt

mkdir -p data
mv *.gctx *.txt data/
```

---

### Step 2 — Preprocess data

```bash
python scripts/01_preprocess_lincs.py
```

Output:

```
level_5_lincs_all_genes.h5ad
```

---

### Step 3 — Create TAS-filtered splits

```bash
python scripts/02_prepare_tas_high_splits.py \
  --adata level_5_lincs_all_genes.h5ad \
  --tas-threshold 0.2 \
  --holdout-cells A549 NCIH508 MDAMB231 LNCAP \
  --outdir splits
```

---

### Step 4 — Train models

```bash
python scripts/03_train_transcriptome_latent_space.py
python scripts/05_train_chemical_latent_space.py
python scripts/06_train_cross_space_mapping.py
```

---

### Step 5 — Evaluate drug prioritization

```bash
python scripts/07_test_drug_prioritisation.py
```
