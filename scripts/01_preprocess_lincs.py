"""
01_preprocess_lincs.py
---------------------

Preprocess LINCS L1000 Level-5 data and create a filtered AnnData object.

This script is a cleaned, GitHub-friendly adaptation of the original research preprocessing script.

Notes
-----
- This script expects local access to the LINCS Level-5 GCTX and the associated metadata
  files (siginfo, geneinfo, compoundinfo).
- It also expects an MOA label sheet (e.g. Huang_MOA_label.xlsx) used to map `pert_id -> MOA`.
- Paths are project-specific; adjust the `DATA_DIR` and filenames as needed.
"""

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse


def main() -> None:
    # ---- Configure paths (edit for your environment)
    repo_root = Path(__file__).resolve().parents[4]
    data_dir = repo_root / "data"
    out_path = Path("level_5_lincs_all_genes.h5ad")

    gctx_path = data_dir / "level5_beta_trt_cp_n720216x12328.gctx"
    siginfo_path = data_dir / "siginfo_beta.txt"
    geneinfo_path = data_dir / "geneinfo_beta.txt"
    compoundinfo_path = data_dir / "compoundinfo_beta.txt"
    moa_xlsx_path = Path("Huang_MOA_label.xlsx")

    # ---- Load expression matrix (GCTX)
    gctx = parse(str(gctx_path))
    lincs = gctx.data_df  # genes x signatures

    # ---- Load signature metadata and align to expression columns
    info = pd.read_csv(siginfo_path, sep="\t")
    info["sig_id"] = pd.Categorical(info["sig_id"], categories=list(lincs.columns), ordered=True)
    info = info.sort_values("sig_id").set_index("sig_id")

    # ---- Load gene metadata, keep landmark+inferred (default in original script)
    geneinfo = pd.read_csv(geneinfo_path, sep="\t")
    geneinfo.index = geneinfo["gene_symbol"]

    lincs.index = lincs.index.astype(int)
    lincs = lincs.loc[geneinfo["gene_id"]].T  # signatures x genes

    # Replace gene ids with gene symbols
    rename_dict = dict(zip(geneinfo["gene_id"], geneinfo.index))
    lincs.rename(columns=rename_dict, inplace=True)

    # ---- Keep only signatures present in metadata
    common = lincs.index.intersection(info.index)
    lincs = lincs.loc[common]
    info = info.loc[common]

    adata = ad.AnnData(X=lincs, obs=info, var=geneinfo)

    # ---- Merge compound canonical SMILES + moa strings from compoundinfo
    comp = pd.read_csv(compoundinfo_path, sep="\t").drop_duplicates(subset="pert_id")
    comp = comp.dropna(subset=["canonical_smiles"]).drop_duplicates(subset="pert_id")
    adata = adata[adata.obs["pert_id"].isin(comp["pert_id"].unique())].copy()
    adata.obs = adata.obs.merge(comp[["pert_id", "moa", "canonical_smiles"]], on="pert_id", how="left")

    # ---- Map pert_id -> curated MOA label (Huang sheet)
    moa_info = pd.read_excel(moa_xlsx_path, header=1).rename(columns={"BRD-ID": "pert_id"})
    adata.obs["pert_id"] = adata.obs["pert_id"].astype(str).str.strip()
    moa_info["pert_id"] = moa_info["pert_id"].astype(str).str.strip()
    moa_map = dict(zip(moa_info["pert_id"], moa_info["MOA"]))
    adata.obs["MOA"] = adata.obs["pert_id"].map(moa_map)

    # ---- Filters (mirrors original)
    # at least 10 samples per drug
    counts = adata.obs["pert_id"].value_counts()
    keep_drugs = counts[counts >= 10].index
    adata = adata[adata.obs["pert_id"].isin(keep_drugs)].copy()

    # hi-quality only
    if "is_hiq" in adata.obs:
        adata = adata[adata.obs["is_hiq"] == 1].copy()

    # keep 6h and 24h
    if "pert_time" in adata.obs:
        adata = adata[adata.obs["pert_time"].isin([6, 24])].copy()

    # ignore restricted SMILES
    adata = adata[adata.obs["canonical_smiles"] != "restricted"].copy()

    # keep only labeled MOAs
    adata = adata[adata.obs["MOA"].notna()].copy()

    # keep MOAs with >= 10 unique drugs
    moa_to_drugs = adata.obs.groupby("MOA")["pert_id"].nunique()
    keep_moas = moa_to_drugs[moa_to_drugs >= 10].index
    adata = adata[adata.obs["MOA"].isin(keep_moas)].copy()

    # Clean var (original dropped gene_symbol column after assigning index)
    if "gene_symbol" in adata.var.columns:
        adata.var = adata.var.drop(columns=["gene_symbol"])

    adata.write(str(out_path))
    print(f"Saved filtered AnnData to {out_path}")


if __name__ == "__main__":
    main()

