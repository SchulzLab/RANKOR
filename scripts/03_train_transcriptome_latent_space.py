"""
03_train_transcriptome_latent_space.py
--------------------------------------

Train the transcriptome encoder (hierarchical ArcFace) on TAS-high splits.

This is a minimally cleaned copy of the original research training script.

It still follows the original research structure (per-fold training, fixed architecture).
"""

import os

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning.losses import ArcFaceLoss
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from typing import Optional


class PerClassArcFaceLoss(nn.Module):
    """ArcFace loss with a per-class margin vector."""

    def __init__(self, num_classes: int, embedding_size: int, margin_vector: torch.Tensor, scale: float = 30.0):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.margin_vector = margin_vector
        self.W = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=1)
        cosine = torch.matmul(embeddings, W.t())
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        per_sample_margin = self.margin_vector[labels].to(embeddings.device)
        phi = torch.cos(theta + per_sample_margin.unsqueeze(1))
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale
        return F.cross_entropy(logits, labels)


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


class HierarchicalArcFace(nn.Module):
    def __init__(
        self,
        hiddensize: int,
        gene_num: int,
        num_moa: int,
        num_cmap: int,
        margin_moa_vector: torch.Tensor,
        margin_cmap: float,
    ):
        super().__init__()
        self.encoder = RepresentModel(hiddensize, gene_num)
        self.arcface_moa = PerClassArcFaceLoss(num_classes=num_moa, embedding_size=hiddensize, margin_vector=margin_moa_vector, scale=30)
        self.arcface_cmap = ArcFaceLoss(num_classes=num_cmap, embedding_size=hiddensize, margin=margin_cmap, scale=30)

    def forward(self, x: torch.Tensor, labels_moa: Optional[torch.Tensor] = None, labels_cmap: Optional[torch.Tensor] = None):
        embeddings = self.encoder(x)
        loss_moa = self.arcface_moa(embeddings, labels_moa) if labels_moa is not None else None
        loss_cmap = self.arcface_cmap(embeddings, labels_cmap) if labels_cmap is not None else None
        return embeddings, loss_moa, loss_cmap


class DrugDataset(Dataset):
    def __init__(self, X: np.ndarray, y_moa: np.ndarray, y_cmap: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_moa = torch.tensor(y_moa, dtype=torch.long)
        self.y_cmap = torch.tensor(y_cmap, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_moa[idx], self.y_cmap[idx]


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    epochs: int = 500,
    patience: int = 30,
    lambda_cmap: float = 0.1,
) -> dict:
    model.train()
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        total_loss = 0.0
        for x, labels_moa, labels_cmap in dataloader:
            x = x.to(device)
            labels_moa = labels_moa.to(device)
            labels_cmap = labels_cmap.to(device)

            optimizer.zero_grad()
            _, loss_moa, loss_cmap = model(x, labels_moa, labels_cmap)
            loss_total = loss_moa + lambda_cmap * loss_cmap
            loss_total.backward()
            optimizer.step()
            total_loss += float(loss_total.item())

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.8f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    return best_state


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_size = 256
    batch_size = 512
    num_epochs = 500
    patience = 30

    os.makedirs("models", exist_ok=True)
    print("Training Hierarchical ArcFace transcriptome encoder (TAS-high).")

    for fold in range(4, 8):
        print(f"\n==== Fold {fold} ====")
        train_file = f"splits/train_TAS_high_fold{fold}.h5ad"
        adata_train = sc.read(train_file)
        adata_train.var_names_make_unique()

        X = adata_train.X.astype(np.float32)
        moa_labels = adata_train.obs["MOA"].astype(str).values
        cmap_labels = adata_train.obs["cmap_name"].astype(str).values

        moa_encoder = LabelEncoder()
        cmap_encoder = LabelEncoder()
        y_moa = moa_encoder.fit_transform(moa_labels)
        y_cmap = cmap_encoder.fit_transform(cmap_labels)

        num_moa = len(np.unique(y_moa))
        num_cmap = len(np.unique(y_cmap))

        # Per-class margins: larger margin for easier classes (as in original file).
        easy_classes = [
            "HDAC-i",
            "TKI",
            "PI3K-i",
            "Topo-i",
            "mTOR-i",
            "HSP-i",
            "MEK/ERK-i",
            "CDK-i",
            "Aurora inh.",
            "EGFR-i",
            "antimicrotubule",
            "antimetabolite",
            "glucocorticoid",
            "antipsychotic",
            "anthelmintic",
            "JAK-i",
            "retinoid",
            "PARP-i",
            "antibiotic",
            "cardiac glycoside",
        ]
        margin_moa_vector = torch.full((num_moa,), 0.25, dtype=torch.float32)
        for c in easy_classes:
            if c in moa_encoder.classes_:
                idx = np.where(moa_encoder.classes_ == c)[0][0]
                margin_moa_vector[idx] = 0.7
        margin_moa_vector = margin_moa_vector.to(device)

        dataset = DrugDataset(X, y_moa, y_cmap)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = HierarchicalArcFace(
            hiddensize=embedding_size,
            gene_num=X.shape[1],
            num_moa=num_moa,
            num_cmap=num_cmap,
            margin_moa_vector=margin_moa_vector,
            margin_cmap=0.1,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        best_state = train_model(
            model,
            dataloader,
            optimizer,
            device,
            epochs=num_epochs,
            patience=patience,
            lambda_cmap=0.1,
        )

        save_path = f"models/nn_hierarcface_high_fold{fold}_tas_high_adaptive_margin.pth"
        torch.save(best_state, save_path)
        print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()

