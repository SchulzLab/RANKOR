import numpy as np
from typing import Optional


def is_valid_smiles(smiles: Optional[str]) -> bool:
    """Return True if SMILES is non-restricted and RDKit-parsable."""
    if smiles is None:
        return False
    s = str(smiles)
    if s.strip().lower() == "restricted":
        return False
    from rdkit import Chem

    return Chem.MolFromSmiles(s) is not None


def morgan_fp(smiles: str, n_bits: int = 256, radius: int = 1) -> np.ndarray:
    """Compute a Morgan fingerprint bit vector (0/1) as a numpy array."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    # RDKit ExplicitBitVect supports numpy conversion directly in many setups; be explicit here.
    return np.array(fp, dtype=np.int8)

