import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import List

from .fingerprints import is_valid_smiles, morgan_fp
from .io import align_profile_to_genes
from .models import DrugEncoder, ExpressionEncoder, LatentToDrugNet


def _load_expression_encoder_from_hierarcface_ckpt(
    *,
    ckpt_path: str,
    gene_num: int,
    latent_dim: int,
    device: torch.device,
) -> ExpressionEncoder:
    """Load only the encoder weights from a hierarchical ArcFace checkpoint."""
    expr_encoder = ExpressionEncoder(gene_num=gene_num, latent_dim=latent_dim).to(device)
    raw = torch.load(ckpt_path, map_location=device)

    # The training code stores encoder weights under "encoder.encoder.*"
    mapped = {}
    for k, v in raw.items():
        if k.startswith("encoder.encoder."):
            mapped[k.replace("encoder.encoder.", "encoder.")] = v
    expr_encoder.load_state_dict(mapped, strict=True)
    expr_encoder.eval()
    return expr_encoder


def _load_drug_encoder_from_arcface_ckpt(
    *,
    ckpt_path: str,
    device: torch.device,
    input_dim: int = 256,
    latent_dim: int = 256,
) -> DrugEncoder:
    """Load the drug encoder weights from an ArcFace training checkpoint."""
    drug_encoder = DrugEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["encoder"] if isinstance(ckpt, dict) and "encoder" in ckpt else ckpt
    drug_encoder.load_state_dict(state, strict=True)
    drug_encoder.eval()
    return drug_encoder


def rank_drugs_from_profile(
    *,
    profile: pd.Series,
    train_h5ad: str,
    expression_encoder_ckpt: str,
    drug_encoder_ckpt: str,
    regressor_ckpt: str,
    topk: int = 0,
    latent_dim: int = 256,
) -> pd.DataFrame:
    """Rank drugs for an input gene profile using trained RANKOR checkpoints.

    Parameters
    ----------
    profile:
        pd.Series indexed by gene symbol with expression values (can be partial).
    train_h5ad:
        Reference AnnData used to define gene order (var_names) and drug library (obs).
    expression_encoder_ckpt:
        Transcriptome encoder checkpoint (hierarchical ArcFace .pth).
    drug_encoder_ckpt:
        Chemical encoder checkpoint (ArcFace encoder .pth).
    regressor_ckpt:
        Cross-space mapping model checkpoint (LatentToDrugNet state_dict .pth).
    topk:
        If > 0, return only the top-k ranked drugs. If 0, return all.
    latent_dim:
        Latent dimension for both spaces (default 256).

    Returns
    -------
    DataFrame with columns: rank, drug_name, cosine_similarity
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import heavier deps lazily to keep module import lightweight.
    import scanpy as sc
    from sklearn.metrics.pairwise import cosine_similarity

    adata = sc.read(train_h5ad)
    adata.var_names_make_unique()
    genes = list(adata.var_names)  # type: List[str]

    # Build the drug library from the reference AnnData
    drug_df = adata.obs[["cmap_name", "canonical_smiles"]].drop_duplicates()
    drug_df = drug_df[drug_df["canonical_smiles"].apply(is_valid_smiles)]
    if drug_df.empty:
        raise ValueError("No valid drugs found in train_h5ad (after SMILES filtering).")

    fps = np.vstack([morgan_fp(s) for s in drug_df["canonical_smiles"].values]).astype(np.float32)
    drug_names = drug_df["cmap_name"].astype(str).values

    drug_encoder = _load_drug_encoder_from_arcface_ckpt(
        ckpt_path=drug_encoder_ckpt, device=device, input_dim=fps.shape[1], latent_dim=latent_dim
    )
    with torch.no_grad():
        drug_emb = F.normalize(drug_encoder(torch.tensor(fps).to(device)), dim=-1).cpu().numpy()

    # Input profile -> aligned vector
    x = align_profile_to_genes(profile, genes).values.astype(np.float32)

    expr_encoder = _load_expression_encoder_from_hierarcface_ckpt(
        ckpt_path=expression_encoder_ckpt,
        gene_num=len(genes),
        latent_dim=latent_dim,
        device=device,
    )

    regressor = LatentToDrugNet(input_dim=latent_dim, output_dim=drug_emb.shape[1]).to(device)
    regressor.load_state_dict(torch.load(regressor_ckpt, map_location=device), strict=True)
    regressor.eval()

    with torch.no_grad():
        latent = expr_encoder(torch.tensor(x[None, :]).to(device))
        pred = regressor(latent).cpu().numpy()

    sims = cosine_similarity(pred, drug_emb).ravel()
    order = np.argsort(-sims)
    if topk and topk > 0:
        order = order[:topk]

    out = pd.DataFrame({"drug_name": drug_names[order], "cosine_similarity": sims[order]})
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

