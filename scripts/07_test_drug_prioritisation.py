"""
07_test_drug_prioritisation.py
------------------------------

Evaluate drug prioritisation on a test split by:
- encoding transcriptome profiles to latent space
- mapping to predicted chemical space via regressor
- ranking drug library by cosine similarity
- exporting per-signature ranking results

This is a minimally cleaned adaptation of the original research evaluation script.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_valid_smiles(s: str) -> bool:
    return s != "restricted" and Chem.MolFromSmiles(s) is not None


def create_fingerprints(df: pd.DataFrame, *, n_bits: int = 256, radius: int = 1) -> pd.DataFrame:
    fps, names = [], []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is None:
            continue
        bitstring = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits).ToBitString()
        fps.append([int(b) for b in bitstring])
        names.append(row["cmap_name"])
    return pd.DataFrame(fps, index=names)


class DrugEncoder(nn.Module):
    def __init__(self, input_dim: int = 256, latent_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_arcface_embeddings(drug_fp: pd.DataFrame, arcface_model_path: str) -> pd.DataFrame:
    encoder = DrugEncoder(input_dim=drug_fp.shape[1], latent_dim=256).to(device)
    state_dict = torch.load(arcface_model_path, map_location=device)
    encoder_state = state_dict["encoder"] if isinstance(state_dict, dict) and "encoder" in state_dict else state_dict
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval()

    with torch.no_grad():
        X = torch.tensor(drug_fp.values, dtype=torch.float32).to(device)
        z = F.normalize(encoder(X), dim=-1).cpu().numpy()
    out = drug_fp.copy()
    out.iloc[:, :] = z
    return out


class RepresentModel(nn.Module):
    def __init__(self, hiddensize: int, gene_num: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(gene_num, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, hiddensize),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LatentToDrugNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_latent_embeddings(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_tensor).cpu().numpy()


def main() -> None:
    all_fold_results = []

    for fold in range(1, 4):
        print(f"\n=== Fold {fold} ===")
        train_file = f"splits/train_TAS_high_fold{fold}.h5ad"
        test_file = f"splits/test_TAS_high_fold{fold}.h5ad"

        adata_train = sc.read(train_file)
        adata_test = sc.read(test_file)
        adata_train.var_names_make_unique()
        adata_test.var_names_make_unique()

        # Drug library from train+test (unique by cmap_name)
        train_smiles = adata_train.obs[["cmap_name", "canonical_smiles"]].drop_duplicates("cmap_name")
        test_smiles = adata_test.obs[["cmap_name", "canonical_smiles"]].drop_duplicates("cmap_name")
        all_drugs = pd.concat([train_smiles, test_smiles]).drop_duplicates("cmap_name")
        all_drugs = all_drugs[all_drugs["canonical_smiles"].apply(is_valid_smiles)]

        drug_fp = create_fingerprints(all_drugs)
        drug_emb = compute_arcface_embeddings(drug_fp, f"arcface_model_{fold}.pth")

        # Transcriptome encoder
        state_dict = torch.load(f"models/nn_hierarcface_high_fold{fold}_tas_high_adaptive_margin.pth", map_location=device)
        enc_state = {k.replace("encoder.encoder.", "encoder."): v for k, v in state_dict.items() if k.startswith("encoder.encoder.")}
        expr_encoder = RepresentModel(hiddensize=256, gene_num=adata_train.X.shape[1]).to(device)
        expr_encoder.load_state_dict(enc_state, strict=False)

        latent_train = compute_latent_embeddings(expr_encoder, adata_train.X.astype(np.float32))
        latent_test = compute_latent_embeddings(expr_encoder, adata_test.X.astype(np.float32))

        # Regressor
        regressor = LatentToDrugNet(input_dim=latent_train.shape[1], output_dim=drug_emb.shape[1]).to(device)
        regressor.load_state_dict(torch.load(f"drug_prioritisation_fold{fold}.pth", map_location=device), strict=True)
        regressor.eval()

        with torch.no_grad():
            latent_test_pred = regressor(torch.tensor(latent_test, dtype=torch.float32).to(device)).cpu().numpy()

        # MOA prediction from transcriptome embeddings (baseline diagnostic)
        knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        knn.fit(latent_train, list(adata_train.obs["MOA"]))
        pred_moa = knn.predict(latent_test)

        # Rank drugs by cosine similarity in chemical latent space
        cos_sim_matrix = cosine_similarity(latent_test_pred, drug_emb.values)
        all_cmap = list(drug_emb.index)

        rows = []
        for i, actual in enumerate(adata_test.obs["cmap_name"].astype(str).values):
            sims = cos_sim_matrix[i]
            ranked_idx = np.argsort(-sims)
            top_5 = [all_cmap[j] for j in ranked_idx[:5]]
            correct_idxs = [j for j, c in enumerate(all_cmap) if str(c) == str(actual)]
            rank_correct = min([ranked_idx.tolist().index(idx) + 1 for idx in correct_idxs]) if correct_idxs else np.nan

            rows.append(
                {
                    "actual_cmap_name": actual,
                    "cell_iname": adata_test.obs.iloc[i].get("cell_iname", np.nan),
                    "pert_id": adata_test.obs.iloc[i].get("pert_id", np.nan),
                    "true_MOA": adata_test.obs.iloc[i].get("MOA", np.nan),
                    "pred_MOA": pred_moa[i],
                    "dose": adata_test.obs.iloc[i].get("pert_dose", np.nan),
                    "time": adata_test.obs.iloc[i].get("pert_itime", np.nan),
                    "ranking_of_correct_drug": rank_correct,
                    "top_5_ranked_drugs": top_5,
                    "cosine_with_top1": float(sims[ranked_idx[0]]),
                }
            )

        df_fold = pd.DataFrame(rows)
        df_fold["fold"] = fold
        out_path = f"drug_similarity_fold{fold}.csv"
        df_fold.to_csv(out_path, index=False)
        print(f"Saved fold results to {out_path}")
        all_fold_results.append(df_fold)


if __name__ == "__main__":
    main()

