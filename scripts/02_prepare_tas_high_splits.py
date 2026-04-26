"""
02_prepare_tas_high_splits.py
----------------------------

Create TAS-high train/test splits with a stratified *group* split by drug (pert_id),
so that drugs do not overlap between train and test within each MOA category.

This is a cleaned adaptation of the original research split-preparation script.
"""

from pathlib import Path

import anndata as ad
import numpy as np
from typing import Set, Tuple


def data_split_stratified_group(
    adata: ad.AnnData, *, test_size: float, random_state: int
) -> Tuple[ad.AnnData, ad.AnnData]:
    rng = np.random.RandomState(random_state)
    obs = adata.obs.copy()
    moa_groups = obs.groupby("MOA")["pert_id"].unique()

    train_drugs = set()  # type: Set[str]
    test_drugs = set()  # type: Set[str]
    for moa, drugs in moa_groups.items():
        drugs = list(set(drugs))
        rng.shuffle(drugs)
        n_test = max(1, int(len(drugs) * test_size))
        test_split = set(drugs[:n_test])
        train_split = set(drugs[n_test:])
        train_drugs.update(train_split)
        test_drugs.update(test_split)

    overlap = train_drugs & test_drugs
    if overlap:
        raise ValueError(f"Overlap detected in train/test drugs: {sorted(overlap)[:10]}")

    adata_train = adata[adata.obs["pert_id"].isin(train_drugs)].copy()
    adata_test = adata[adata.obs["pert_id"].isin(test_drugs)].copy()
    return adata_train, adata_test


def save_split(fold: int, adata_train: ad.AnnData, adata_test: ad.AnnData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_file = out_dir / f"train_TAS_high_fold{fold}.h5ad"
    test_file = out_dir / f"test_TAS_high_fold{fold}.h5ad"
    adata_train.write(str(train_file))
    adata_test.write(str(test_file))
    print(
        f"Fold {fold}: train {adata_train.n_obs} obs ({adata_train.obs['pert_id'].nunique()} drugs), "
        f"test {adata_test.n_obs} obs ({adata_test.obs['pert_id'].nunique()} drugs)"
    )
    print(f"  saved: {train_file}")
    print(f"  saved: {test_file}")


def main() -> None:
    # Input file produced by 01_preprocess_lincs.py (or your equivalent).
    in_path = Path("level_5_lincs_all_genes.h5ad")
    out_dir = Path("splits")

    adata = ad.read_h5ad(str(in_path))

    # Filter to TAS-high subset (matches original intent).
    if "tas" in adata.obs:
        adata = adata[adata.obs["tas"] > 0.2].copy()
    else:
        raise ValueError("Expected `tas` column in adata.obs to filter TAS-high.")

    # Create a few folds (original file created 3 splits explicitly).
    for fold, seed in [(1, 20), (2, 21), (3, 22)]:
        tr, te = data_split_stratified_group(adata, test_size=0.2, random_state=seed)
        save_split(fold, tr, te, out_dir)


if __name__ == "__main__":
    main()

