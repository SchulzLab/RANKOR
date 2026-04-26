"""
06_train_cross_space_mapping.py
-------------------------------

Train the cross-space mapping model (LatentToDrugNet) that maps:
  transcriptome latent -> chemical latent

This is a cleaned adaptation of the original research training script.

Outputs:
- regressor checkpoint(s): drug_prioritisation_fold{fold}.pth
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import DataLoader, Dataset


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


def compute_latent_embeddings(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_tensor).cpu().numpy()


class LatentToDrugDataset(Dataset):
    def __init__(self, X_latent: np.ndarray, y_drug: np.ndarray):
        self.X = torch.tensor(X_latent, dtype=torch.float32)
        self.y = torch.tensor(y_drug, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


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


def train_model(model: nn.Module, dataloader: DataLoader, *, lr: float = 1e-3, epochs: int = 50) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            # cosine loss (1 - cosine similarity)
            loss = (1 - F.cosine_similarity(y_pred, y, dim=1)).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * X.size(0)
        print(f"Epoch {epoch+1}/{epochs} | Cosine loss: {epoch_loss/len(dataloader.dataset):.6f}")
    return model


def main() -> None:
    all_fold_results = {}

    for fold in range(1, 4):
        print(f"\n=== Fold {fold} ===")
        train_file = f"splits/train_TAS_high_fold{fold}.h5ad"
        adata_train = sc.read(train_file)
        adata_train.var_names_make_unique()

        # Drug library for this fold (unique by cmap_name)
        drug_df = adata_train.obs[["cmap_name", "canonical_smiles"]].drop_duplicates("cmap_name")
        drug_df = drug_df[drug_df["canonical_smiles"].apply(is_valid_smiles)]
        drug_fp = create_fingerprints(drug_df)
        drug_emb = compute_arcface_embeddings(drug_fp, f"arcface_model_{fold}.pth")

        # Transcriptome encoder weights
        state_dict = torch.load(f"models/nn_hierarcface_high_fold{fold}_tas_high_adaptive_margin.pth", map_location=device)
        enc_state = {k.replace("encoder.encoder.", "encoder."): v for k, v in state_dict.items() if k.startswith("encoder.encoder.")}
        expr_encoder = RepresentModel(hiddensize=256, gene_num=adata_train.X.shape[1]).to(device)
        expr_encoder.load_state_dict(enc_state, strict=False)

        latent_train = compute_latent_embeddings(expr_encoder, adata_train.X.astype(np.float32))
        latent_df = pd.DataFrame(latent_train, index=adata_train.obs["cmap_name"].astype(str))

        common = latent_df.index.intersection(drug_emb.index.astype(str))
        latent_df = latent_df.loc[common]
        drug_target = drug_emb.loc[common].values.astype(np.float32)

        dataset = LatentToDrugDataset(latent_df.values.astype(np.float32), drug_target)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        regressor = LatentToDrugNet(input_dim=latent_df.shape[1], output_dim=drug_target.shape[1]).to(device)
        regressor = train_model(regressor, dataloader, lr=1e-3, epochs=50)

        # Quick sanity metric on train: mean cosine similarity
        regressor.eval()
        with torch.no_grad():
            y_pred = regressor(torch.tensor(latent_df.values.astype(np.float32)).to(device)).cpu().numpy()
        mean_cos = float(np.mean((y_pred * drug_target).sum(axis=1) / (np.linalg.norm(y_pred, axis=1) * np.linalg.norm(drug_target, axis=1) + 1e-12)))
        print(f"Fold {fold} | mean cosine similarity (train): {mean_cos:.6f}")

        out_path = f"drug_prioritisation_fold{fold}.pth"
        torch.save(regressor.state_dict(), out_path)
        print(f"Saved regressor to {out_path}")

        all_fold_results[fold] = {"regressor_path": out_path, "mean_cosine_similarity_train": mean_cos}


if __name__ == "__main__":
    main()

