"""
05_train_chemical_latent_space.py
---------------------------------

Train the chemical latent space encoder using Morgan fingerprints.

This is a minimally cleaned copy of the original research training script.

It trains an encoder with an ArcFace classification head plus a triplet loss.
The saved checkpoint is expected to contain at least:
  {"encoder": <state_dict>, "arcface": <state_dict>}
"""

import random

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple


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


class ContrastiveDrugDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.features = df.drop(columns=["MOA"]).values.astype(np.float32)
        self.labels = df["MOA"].astype("category").cat.codes.values.astype(np.int64)
        self.class_to_indices = {}  # type: Dict[int, List[int]]
        for idx, label in enumerate(self.labels):
            self.class_to_indices.setdefault(int(label), []).append(idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        X = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        # anchor/positive/negative for triplet loss
        anchor = X
        pos_candidates = [i for i in self.class_to_indices[int(y.item())] if i != idx]
        positive_idx = random.choice(pos_candidates) if pos_candidates else idx
        negative_class = random.choice([c for c in self.class_to_indices.keys() if c != int(y.item())])
        negative_idx = random.choice(self.class_to_indices[negative_class])

        positive = torch.tensor(self.features[positive_idx], dtype=torch.float32)
        negative = torch.tensor(self.features[negative_idx], dtype=torch.float32)
        return X, y, anchor, positive, negative


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


class ArcFace(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        weights = F.normalize(self.weight, dim=-1)
        cos_theta = torch.matmul(embeddings, weights.t()).clamp(-1, 1)
        theta = torch.acos(cos_theta)
        cos_theta_m = torch.cos(theta + self.margin)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        return logits * self.scale


def train_arcface(
    df: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    save_path: str,
    alpha: float,
    beta: float,
) -> Tuple[DrugEncoder, ArcFace]:
    df = df.copy()
    df["MOA_code"] = df["MOA"].astype("category").cat.codes
    X_cols = [c for c in df.columns if c not in ["MOA", "MOA_code"]]

    dataset = ContrastiveDrugDataset(df[X_cols + ["MOA"]])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    num_classes = int(df["MOA_code"].nunique())
    encoder = DrugEncoder(input_dim=len(X_cols), latent_dim=256).to(device)
    arcface = ArcFace(embedding_dim=256, num_classes=num_classes).to(device)

    ce_loss = nn.CrossEntropyLoss()
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(list(encoder.parameters()) + list(arcface.parameters()), lr=lr)

    for epoch in range(epochs):
        encoder.train()
        arcface.train()
        total_loss = 0.0

        for X, y, anchor, positive, negative in dataloader:
            X, y = X.to(device), y.to(device)
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            embeddings = encoder(X)
            logits = arcface(embeddings, y)
            loss_cls = ce_loss(logits, y)

            emb_anchor = encoder(anchor)
            emb_pos = encoder(positive)
            emb_neg = encoder(negative)
            loss_triplet = triplet_loss(emb_anchor, emb_pos, emb_neg)

            loss = alpha * loss_cls + beta * loss_triplet
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * X.size(0)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataset):.6f}")

    torch.save({"encoder": encoder.state_dict(), "arcface": arcface.state_dict()}, save_path)
    print(f"Saved chemical model to {save_path}")
    return encoder, arcface


def main() -> None:
    # Example: train on fold=1 split files; adjust as needed.
    fold = 1
    train_file = f"splits/train_TAS_high_fold{fold}.h5ad"
    test_file = f"splits/test_TAS_high_fold{fold}.h5ad"

    adata_train = sc.read(train_file)
    adata_test = sc.read(test_file)

    train_unique = adata_train.obs[["cmap_name", "canonical_smiles", "MOA"]].drop_duplicates("cmap_name")
    test_unique = adata_test.obs[["cmap_name", "canonical_smiles", "MOA"]].drop_duplicates("cmap_name")

    all_unique = pd.concat([train_unique, test_unique], axis=0).drop_duplicates("cmap_name")
    all_unique = all_unique[all_unique["canonical_smiles"].apply(is_valid_smiles)]

    drug_fp = create_fingerprints(all_unique)
    drug_fp = drug_fp.merge(all_unique[["cmap_name", "MOA"]], left_index=True, right_on="cmap_name").set_index("cmap_name")

    train_arcface(
        df=drug_fp,
        epochs=80,
        batch_size=64,
        lr=1e-3,
        save_path=f"arcface_model_{fold}.pth",
        alpha=1.0,
        beta=7.0,
    )


if __name__ == "__main__":
    main()

