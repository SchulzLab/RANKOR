"""RANKOR: rank-based cross-space drug prioritization.

This package is designed for GitHub sharing and lightweight importing.
Heavier scientific dependencies (e.g. scanpy, rdkit) are imported lazily.
"""

__all__ = ["rank_drugs_from_profile"]


def rank_drugs_from_profile(**kwargs):
    # Lazy import to avoid importing heavy deps at package import time.
    from .predict import rank_drugs_from_profile as _rank

    return _rank(**kwargs)

