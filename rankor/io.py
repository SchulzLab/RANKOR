import pandas as pd
from typing import List


def read_gene_value_csv(path: str) -> pd.Series:
    """Read a 2-column CSV with columns: gene,value into a Series indexed by gene."""
    df = pd.read_csv(path)
    missing = {"gene", "value"} - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV must contain columns gene,value. Missing: {sorted(missing)}")
    genes = df["gene"].astype(str).values
    values = df["value"].astype(float).values
    return pd.Series(values, index=genes)


def align_profile_to_genes(profile: pd.Series, target_genes: List[str]) -> pd.Series:
    """Align a gene->value Series to the reference gene list (fill missing with 0)."""
    return profile.reindex(target_genes).fillna(0.0).astype("float32")

