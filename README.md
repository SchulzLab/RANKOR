# RANKOR: Direct Drug Prioritization from Bulk and Single-Cell Transcriptomic Signatures

RANKOR is a **two-space framework** that:

1. Learns a **transcriptome latent space** from LINCS L1000 expression signatures (TAS-high subset).
2. Learns a **chemical latent space** from **Morgan fingerprints** (SMILES).
3. Trains a **cross-space mapping** model: transcriptome latent \(\rightarrow\) chemical latent.
4. **Prioritizes drugs** for a user-provided gene profile by ranking drugs via **cosine similarity** in chemical latent space.

---

## Folder structure

```text
  README.md
  README.txt
  requirements.txt
  environment.yml
  RANKOR_prior.py
  rankor/
    __init__.py
    cli.py
    predict.py
    models.py
    fingerprints.py
    io.py
  scripts/
    01_preprocess_lincs.py
    02_prepare_tas_high_splits.py
    03_train_transcriptome_latent_space.py
    04_test_transcriptome_latent_space.py
    05_train_chemical_latent_space.py
    06_train_cross_space_mapping.py
    07_test_drug_prioritisation.py

---

## What you need for inference (drug ranking)

### 1) Input gene profile

Provide a CSV with **exactly** two columns:

```csv
gene,value
STAT1,1.23
IFIT3,0.87
...
```

- Missing genes are allowed; they are filled with **0** after alignment to the reference gene list.

### 2) Reference `.h5ad`

You must provide a reference AnnData file (typically a training fold file) used to define:

- the **gene order** (`adata.var_names`)
- the **drug library** (`adata.obs[["cmap_name","canonical_smiles"]]`)



## Install

### Option A (recommended): conda

From the repository root:

```bash
conda env create -f environment.yml
conda activate rankor
```

### Option B: pip

From the repository root:

```bash
python -m pip install -r equirements.txt
```

---

## Run inference (rank drugs)

From the repository root:

```bash
python -m src.moa.github_rankor.rankor.cli \
  --input path/to/profile.csv \
  --train_h5ad name_of_reference_dataset.h5ad \
  --expression_encoder_ckpt expression_encoder.pth \
  --drug_encoder_ckpt drug_encoder.pth \
  --regressor_ckpt drug_prioritisation.pth \
  --out_csv out_ranked_drugs.csv \
  --topk 100 \
  --print_top 20
```

### Output

The output CSV (`--out_csv`) contains:

- `rank`
- `drug_name`
- `cosine_similarity`

sorted by `cosine_similarity` (descending).


## How inference works (high-level)

1. Load `train_h5ad` and extract:
   - reference gene list
   - drug library (`cmap_name`, `canonical_smiles`)
2. Convert each drug SMILES to a Morgan fingerprint bit-vector.
3. Encode drug fingerprints into **chemical embeddings** with the drug encoder.
4. Read your input gene profile and align to the reference gene list.
5. Encode the profile into a **transcriptome latent** with the expression encoder.
6. Map transcriptome latent \(\rightarrow\) predicted chemical latent with the regressor.
7. Rank drugs by cosine similarity between predicted chemical latent and drug library embeddings.

---

## Training / reproduction

The `scripts/` directory contains pipeline stages extracted from the original research code.
They are intended to be runnable, but **paths and data availability are project-specific**
(LINCS GCTX, metadata files, curated MOA sheet, etc.).

Recommended order:

1. `scripts/01_preprocess_lincs.py`
2. `scripts/02_prepare_tas_high_splits.py`
3. `scripts/03_train_transcriptome_latent_space.py`
4. `scripts/05_train_chemical_latent_space.py`
5. `scripts/06_train_cross_space_mapping.py`
6. `scripts/07_test_drug_prioritisation.py`

---

## Notes / assumptions

- RANKOR assumes the **same gene set and ordering** as the reference `train_h5ad` file.
- Chemical fingerprints are **Morgan radius=1, nBits=256**, matching the original code in `src/moa/`.
- Similarity is **cosine similarity** in the learned chemical latent space.

