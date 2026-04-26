"""
04_test_transcriptome_latent_space.py
-------------------------------------

Evaluate the transcriptome encoder by:
- computing embeddings for train/test
- training a KNN classifier on embeddings
- reporting overall and per-MOA metrics

This is a minimally cleaned version of the original research evaluation script.
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset
from typing import Tuple


class DrugDataset(Dataset):
    def __init__(self, X: np.ndarray, y_moa: np.ndarray, y_cmap: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_moa = torch.tensor(y_moa, dtype=torch.long)
        self.y_cmap = torch.tensor(y_cmap, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_moa[idx], self.y_cmap[idx]


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


def compute_embeddings(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    model.eval()
    embs = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, _, _ in dataloader:
            x = x.to(device)
            embs.append(model(x).cpu())
    return torch.cat(embs).numpy()


def evaluate_fold(fold: int, *, embedding_size: int = 256) -> Tuple[float, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_file = f"splits/train_TAS_high_fold{fold}.h5ad"
    test_file = f"splits/test_TAS_high_fold{fold}.h5ad"
    adata_train = sc.read(train_file)
    adata_test = sc.read(test_file)
    adata_train.var_names_make_unique()
    adata_test.var_names_make_unique()

    X_train = adata_train.X.astype(np.float32)
    X_test = adata_test.X.astype(np.float32)

    # Ensure shared categorical codes across train/test
    all_moa = pd.concat([adata_train.obs["MOA"], adata_test.obs["MOA"]]).astype("category")
    categories = all_moa.cat.categories
    adata_train.obs["MOA"] = adata_train.obs["MOA"].astype(pd.CategoricalDtype(categories=categories))
    adata_test.obs["MOA"] = adata_test.obs["MOA"].astype(pd.CategoricalDtype(categories=categories))

    train_labels = adata_train.obs["MOA"].cat.codes.values
    test_labels = adata_test.obs["MOA"].cat.codes.values

    # Dataloaders (labels not used in embedding computation beyond batch structure)
    y_dummy_train = np.zeros(len(X_train), dtype=np.int64)
    y_dummy_test = np.zeros(len(X_test), dtype=np.int64)
    dataset_train = DrugDataset(X_train, y_dummy_train, y_dummy_train)
    dataset_test = DrugDataset(X_test, y_dummy_test, y_dummy_test)
    dl_train = DataLoader(dataset_train, batch_size=1000, shuffle=False)
    dl_test = DataLoader(dataset_test, batch_size=1000, shuffle=False)

    # Load encoder weights from hierarchical ArcFace checkpoint
    state_dict = torch.load(
        f"models/nn_hierarcface_high_fold{fold}_tas_high_adaptive_margin.pth",
        map_location=device,
    )
    encoder_state = {
        k.replace("encoder.encoder.", "encoder."): v for k, v in state_dict.items() if k.startswith("encoder.encoder.")
    }
    model = RepresentModel(hiddensize=embedding_size, gene_num=X_train.shape[1]).to(device)
    model.load_state_dict(encoder_state, strict=False)

    train_emb = compute_embeddings(model, dl_train)
    test_emb = compute_embeddings(model, dl_test)

    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn.fit(train_emb, train_labels)
    pred_test = knn.predict(test_emb)
    prob_test = knn.predict_proba(test_emb)

    overall_acc = accuracy_score(test_labels, pred_test)

    per_moa_metrics = []
    num_classes = len(categories)
    for i, moa_name in enumerate(categories):
        y_true = (test_labels == i).astype(int)
        y_pred = (pred_test == i).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        support = int(tp + fn)
        total = int(tp + tn + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        accuracy = (tp + tn) / total if total else np.nan
        f1 = (2 * precision * recall / (precision + recall)) if (precision == precision and recall == recall and (precision + recall) > 0) else np.nan

        if support > 0 and support < total:
            roc_auc = roc_auc_score(y_true, prob_test[:, i])
            pr_auc = average_precision_score(y_true, prob_test[:, i])
        else:
            roc_auc = np.nan
            pr_auc = np.nan

        per_moa_metrics.append(
            {
                "MOA": moa_name,
                "Support": support,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Accuracy": accuracy,
                "Specificity": specificity,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc,
            }
        )

    return overall_acc, pd.DataFrame(per_moa_metrics)


def main() -> None:
    os.makedirs("MOA_RESULTS", exist_ok=True)
    results = []

    for fold in range(1, 8):
        acc, df = evaluate_fold(fold, embedding_size=256)
        df["Fold"] = fold
        df.to_csv(f"MOA_RESULTS/RANKOR_{fold}_MOA_predictions.csv", index=False)
        print(f"Fold {fold} | overall accuracy: {acc:.4f}")
        results.append(df)

    _ = pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    main()

