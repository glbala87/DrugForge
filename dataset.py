"""DrugForge data processing — fully original featurization pipeline.

Atom descriptors: 17 continuous features from RDKit
Bond descriptors: 6 continuous features
Protein encoding: 8-dim physicochemical OR ESM-2 1280-dim pre-trained embeddings
Drug features: GIN graph + optional Morgan fingerprints
Auxiliary targets: RDKit molecular descriptors + functional group labels
Graph construction: direct from RDKit bonds
Dataset: simple list-based PyG-compatible dataset
"""

import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import Fragments
from torch_geometric import data as DATA
from tqdm.auto import tqdm

from tokenizer import MolTokenizer

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Atom descriptors — 17 continuous features
# ═══════════════════════════════════════════════════════════════════════════

_HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.S: 0,
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
}


def compute_atom_descriptors(atom) -> np.ndarray:
    """Compute 17 continuous atomic descriptors.

    Features:
        0: atomic number / 53
        1: degree / 6
        2: total valence / 8
        3: num hydrogens / 8
        4: formal charge (shifted: +3 then /6)
        5: radical electrons / 3
        6: hybridization index / 5
        7: is aromatic
        8: is in ring
        9: smallest ring size / 12 (0 if not in ring)
        10: in 3-membered ring
        11: in 5-membered ring
        12: in 6-membered ring
        13: atomic mass / 200
        14: implicit valence / 6
        15: number of heavy-atom neighbors / 6
        16: is chiral center
    """
    ring_info = atom.GetOwningMol().GetRingInfo()
    atom_idx = atom.GetIdx()

    # Smallest ring this atom participates in
    min_ring = 0
    if atom.IsInRing():
        for size in range(3, 13):
            if ring_info.IsAtomInRingOfSize(atom_idx, size):
                min_ring = size
                break

    features = np.array([
        atom.GetAtomicNum() / 53.0,
        atom.GetDegree() / 6.0,
        atom.GetTotalValence() / 8.0,
        atom.GetTotalNumHs() / 8.0,
        (atom.GetFormalCharge() + 3) / 6.0,
        atom.GetNumRadicalElectrons() / 3.0,
        _HYBRIDIZATION_MAP.get(atom.GetHybridization(), 0) / 5.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        min_ring / 12.0,
        float(ring_info.IsAtomInRingOfSize(atom_idx, 3)),
        float(ring_info.IsAtomInRingOfSize(atom_idx, 5)),
        float(ring_info.IsAtomInRingOfSize(atom_idx, 6)),
        atom.GetMass() / 200.0,
        atom.GetValence(Chem.ValenceType.IMPLICIT) / 6.0 if hasattr(Chem, 'ValenceType') else atom.GetImplicitValence() / 6.0,
        len([n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]) / 6.0,
        float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
    ], dtype=np.float32)

    return features


# ═══════════════════════════════════════════════════════════════════════════
# Bond descriptors — 6 continuous features
# ═══════════════════════════════════════════════════════════════════════════

_STEREO_MAP = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOZ: 1,
    Chem.rdchem.BondStereo.STEREOE: 2,
    Chem.rdchem.BondStereo.STEREOCIS: 3,
    Chem.rdchem.BondStereo.STEREOTRANS: 4,
    Chem.rdchem.BondStereo.STEREOANY: 5,
}


def compute_bond_descriptors(bond) -> np.ndarray:
    """Compute 6 continuous bond descriptors.

    Features:
        0: bond order (1.0 / 1.5 / 2.0 / 3.0) normalized by 3
        1: is conjugated
        2: is in ring
        3: stereo type / 5
        4: is aromatic (from bond type)
        5: bond order squared / 9 (captures non-linearity)
    """
    order = bond.GetBondTypeAsDouble()
    return np.array([
        order / 3.0,
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        _STEREO_MAP.get(bond.GetStereo(), 0) / 5.0,
        float(bond.GetIsAromatic()),
        (order * order) / 9.0,
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# SMILES → molecular graph (direct from RDKit bond iteration)
# ═══════════════════════════════════════════════════════════════════════════

def smiles_to_graph(smiles: str) -> Optional[dict]:
    """Convert SMILES to graph dict with atom/bond features.

    Returns dict with keys: x, edge_index, edge_attr, num_atoms
    or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_feats = np.array(
        [compute_atom_descriptors(a) for a in mol.GetAtoms()],
        dtype=np.float32,
    )

    # Edge features — build directed adjacency directly from RDKit bonds
    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = compute_bond_descriptors(bond)
        # Both directions for undirected graph
        src.extend([i, j])
        dst.extend([j, i])
        edge_feats.extend([bf, bf])

    if len(src) == 0:
        # Single atom molecule — add self-loop
        src, dst = [0], [0]
        edge_feats = [np.zeros(6, dtype=np.float32)]

    return {
        'x': atom_feats,
        'edge_index': np.array([src, dst], dtype=np.int64),
        'edge_attr': np.array(edge_feats, dtype=np.float32),
        'num_atoms': mol.GetNumAtoms(),
    }


def build_graph_cache(smiles_list: List[str]) -> Dict[str, dict]:
    """Pre-compute molecular graphs for all unique SMILES."""
    cache = {}
    for smi in tqdm(set(smiles_list), desc='Building graph cache'):
        g = smiles_to_graph(smi)
        if g is not None:
            cache[smi] = g
    return cache


# ═══════════════════════════════════════════════════════════════════════════
# Protein encoding — 8 physicochemical properties per residue
# ═══════════════════════════════════════════════════════════════════════════

# Kyte-Doolittle hydrophobicity (normalized to approx [-1, 1])
_HYDRO = {
    'A': 0.40, 'R': -1.0, 'N': -0.78, 'D': -0.78, 'C': 0.56,
    'Q': -0.78, 'E': -0.78, 'G': -0.09, 'H': -0.71, 'I': 1.0,
    'L': 0.84, 'K': -0.87, 'M': 0.42, 'F': 0.62, 'P': -0.36,
    'S': -0.18, 'T': -0.16, 'W': -0.20, 'Y': -0.29, 'V': 0.93,
}

# Net charge at pH 7
_CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1,
    'G': 0, 'H': 0.1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
    'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}

# Normalized molecular weight (divided by max ~204)
_WEIGHT = {
    'A': 0.44, 'R': 0.85, 'N': 0.65, 'D': 0.65, 'C': 0.59,
    'Q': 0.72, 'E': 0.72, 'G': 0.37, 'H': 0.76, 'I': 0.64,
    'L': 0.64, 'K': 0.72, 'M': 0.73, 'F': 0.81, 'P': 0.56,
    'S': 0.51, 'T': 0.58, 'W': 1.0, 'Y': 0.89, 'V': 0.57,
}

# Polarity (1=polar, 0=nonpolar)
_POLAR = {
    'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0, 'Q': 1, 'E': 1,
    'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
    'P': 0, 'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0,
}

# H-bond donor capacity
_HBD = {
    'A': 0, 'R': 1, 'N': 1, 'D': 0, 'C': 0, 'Q': 1, 'E': 0,
    'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
    'P': 0, 'S': 1, 'T': 1, 'W': 1, 'Y': 1, 'V': 0,
}

# H-bond acceptor capacity
_HBA = {
    'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0, 'Q': 1, 'E': 1,
    'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 0, 'M': 0, 'F': 0,
    'P': 0, 'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0,
}

# Aromatic side chain
_AROM = {
    'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0, 'Q': 0, 'E': 0,
    'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 0, 'M': 0, 'F': 1,
    'P': 0, 'S': 0, 'T': 0, 'W': 1, 'Y': 1, 'V': 0,
}

# Normalized van der Waals volume (divided by max ~228)
_VDW = {
    'A': 0.39, 'R': 0.73, 'N': 0.50, 'D': 0.47, 'C': 0.47,
    'Q': 0.56, 'E': 0.54, 'G': 0.26, 'H': 0.66, 'I': 0.67,
    'L': 0.67, 'K': 0.68, 'M': 0.67, 'F': 0.77, 'P': 0.51,
    'S': 0.39, 'T': 0.50, 'W': 1.0, 'Y': 0.83, 'V': 0.58,
}

_PROPERTY_TABLES = [_HYDRO, _CHARGE, _WEIGHT, _POLAR, _HBD, _HBA, _AROM, _VDW]
_DEFAULT_RESIDUE = np.zeros(8, dtype=np.float32)

PROTEIN_MAX_LEN = 1200


def encode_protein(sequence: str) -> np.ndarray:
    """Encode protein as (max_len, 8) physicochemical property matrix.

    Each residue is represented by 8 continuous features:
        hydrophobicity, charge, weight, polarity,
        H-bond donor, H-bond acceptor, aromaticity, vdW volume
    """
    encoded = np.zeros((PROTEIN_MAX_LEN, 8), dtype=np.float32)
    for i, aa in enumerate(sequence[:PROTEIN_MAX_LEN]):
        for j, table in enumerate(_PROPERTY_TABLES):
            encoded[i, j] = table.get(aa, 0.0)
    return encoded


# ═══════════════════════════════════════════════════════════════════════════
# Morgan fingerprint features (supplementary drug representation)
# ═══════════════════════════════════════════════════════════════════════════

def compute_morgan_fp(smiles: str, n_bits: int = 256, radius: int = 2) -> np.ndarray:
    """Compute Morgan (circular) fingerprint as a fixed-length bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Molecular descriptor targets (auxiliary regression task)
# ═══════════════════════════════════════════════════════════════════════════

# Collect all calculable RDKit descriptors
_DESCRIPTOR_FUNCS = [(name, func) for name, func in Descriptors._descList]
_NUM_DESCRIPTORS = len(_DESCRIPTOR_FUNCS)


def compute_mol_descriptors(smiles: str) -> np.ndarray:
    """Compute all RDKit molecular descriptors for a SMILES string.

    Returns a vector of ~120 descriptor values (NaN replaced with 0).
    Descriptors are NOT normalized here — normalization happens during training.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(_NUM_DESCRIPTORS, dtype=np.float32)

    desc = []
    for name, func in _DESCRIPTOR_FUNCS:
        try:
            val = func(mol)
            desc.append(float(val) if np.isfinite(val) else 0.0)
        except Exception:
            desc.append(0.0)
    return np.array(desc, dtype=np.float32)


def compute_descriptor_stats(smiles_list):
    """Compute min/max statistics for descriptor normalization."""
    all_desc = []
    for smi in tqdm(set(smiles_list), desc='Computing descriptor stats'):
        all_desc.append(compute_mol_descriptors(smi))
    arr = np.array(all_desc)
    mins = np.nanmin(arr, axis=0)
    maxs = np.nanmax(arr, axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # avoid division by zero
    return mins, ranges


def normalize_descriptors(desc: np.ndarray, mins: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Min-max normalize descriptors to [0, 1]."""
    return np.clip((desc - mins) / ranges, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Functional group labels (auxiliary multi-label classification task)
# ═══════════════════════════════════════════════════════════════════════════

# Collect all RDKit fragment counter functions (fr_* functions)
_FRAG_FUNCS = [(name, func) for name, func in Descriptors._descList
               if name.startswith('fr_')]
_NUM_FRAG_FUNCS = len(_FRAG_FUNCS)


def compute_functional_groups(smiles: str) -> np.ndarray:
    """Compute binary functional group presence vector.

    Uses RDKit's fr_* fragment descriptor functions to detect
    presence of ~85 functional groups/motifs in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(_NUM_FRAG_FUNCS, dtype=np.float32)

    labels = []
    for name, func in _FRAG_FUNCS:
        try:
            count = func(mol)
            labels.append(1.0 if count > 0 else 0.0)
        except Exception:
            labels.append(0.0)
    return np.array(labels, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# ESM-2 protein embeddings (optional, requires fair-esm)
# ═══════════════════════════════════════════════════════════════════════════

def extract_esm2_embeddings(sequences, model_name='esm2_t33_650M_UR50D',
                            repr_layer=33, device='cpu', batch_size=4):
    """Extract protein embeddings using ESM-2 language model.

    Args:
        sequences: list of (id, sequence) tuples
        model_name: ESM-2 model variant
        repr_layer: which transformer layer to extract from
        device: 'cpu' or 'cuda'
        batch_size: batch size for inference

    Returns:
        dict mapping sequence_id -> np.ndarray of shape (esm2_dim,)
    """
    try:
        import esm
    except ImportError:
        raise ImportError(
            "ESM-2 requires the `fair-esm` package. Install with:\n"
            "  pip install fair-esm\n"
            "Or set use_esm2=False in ModelConfig."
        )

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    embeddings = {}
    for i in tqdm(range(0, len(sequences), batch_size), desc='Extracting ESM-2'):
        batch_seqs = sequences[i:i + batch_size]
        # Truncate to ESM-2 max length (1022 residues)
        batch_seqs = [(sid, seq[:1022]) for sid, seq in batch_seqs]

        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seqs)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer])
        reps = results['representations'][repr_layer]  # (B, L, D)

        for j, (sid, seq) in enumerate(batch_seqs):
            # Mean pool over residue positions (exclude BOS/EOS tokens)
            seq_len = len(seq)
            token_rep = reps[j, 1:seq_len + 1]  # skip BOS token
            embeddings[sid] = token_rep.mean(dim=0).cpu().numpy()

    return embeddings


# ═══════════════════════════════════════════════════════════════════════════
# Cold-start data splitting (drug/protein cold-start evaluation)
# ═══════════════════════════════════════════════════════════════════════════

def cold_start_split(df, entity_col, n_folds=5, seed=42):
    """Split data for cold-start evaluation.

    Ensures that entities in test set are NEVER seen during training.

    Args:
        df: DataFrame with drug-target pairs
        entity_col: column to split on ('compound_iso_smiles' for drug cold-start,
                    'target_sequence' for protein cold-start)
        n_folds: number of cross-validation folds
        seed: random seed

    Returns:
        list of (train_df, test_df) tuples
    """
    rng = np.random.RandomState(seed)
    entities = df[entity_col].unique()
    rng.shuffle(entities)

    fold_size = len(entities) // n_folds
    folds = []

    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        if fold_idx == n_folds - 1:
            test_entities = set(entities[start:])
        else:
            test_entities = set(entities[start:start + fold_size])

        test_mask = df[entity_col].isin(test_entities)
        train_df = df[~test_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)
        folds.append((train_df, test_df))

    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Dataset — simple list-based PyG-compatible dataset
# ═══════════════════════════════════════════════════════════════════════════

class MolProteinDataset:
    """Simple dataset of PyG Data objects for drug-target pairs.

    Stores pre-processed graphs on disk and loads them lazily.
    Compatible with torch_geometric.loader.DataLoader.
    """

    def __init__(self, data_list: Optional[List[DATA.Data]] = None,
                 cache_path: Optional[str] = None):
        if cache_path and os.path.isfile(cache_path):
            logger.info(f'Loading dataset from {cache_path}')
            # weights_only=False required for PyG Data objects with custom attributes
            self.data_list = torch.load(cache_path, weights_only=False)  # nosec: trusted local cache
        elif data_list is not None:
            self.data_list = data_list
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(data_list, cache_path)
                logger.info(f'Dataset saved to {cache_path}')
        else:
            raise ValueError('Provide either data_list or a valid cache_path')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def build_dataset(drugs, drug_tokens, proteins, affinities, graph_cache,
                  cache_path=None, morgan_fp_dim=256,
                  include_descriptors=True, include_funcgroups=True,
                  desc_stats=None, labels=None):
    """Build a MolProteinDataset from raw data arrays.

    Args:
        drugs: list of SMILES strings
        drug_tokens: list of LongTensor token sequences
        proteins: ndarray of shape (N, max_len, 8) or (N, esm2_dim) for ESM-2
        affinities: list/array of float values (affinity or binary labels)
        graph_cache: dict mapping SMILES -> graph dict
        cache_path: optional path to save/load processed dataset
        morgan_fp_dim: number of Morgan fingerprint bits (0 to skip)
        include_descriptors: whether to compute molecular descriptor targets
        include_funcgroups: whether to compute functional group labels
        desc_stats: (mins, ranges) tuple for descriptor normalization
        labels: optional dict with 'dti' and/or 'moa' binary labels
    """
    if cache_path and os.path.isfile(cache_path):
        return MolProteinDataset(cache_path=cache_path)

    # Pad token sequences
    pad_id = MolTokenizer.CONTROL_TOKENS.index('[PAD]')
    max_tok_len = max(len(t) for t in drug_tokens)
    padded_tokens = []
    for t in drug_tokens:
        padded = torch.full((max_tok_len,), pad_id, dtype=torch.long)
        padded[:len(t)] = t
        padded_tokens.append(padded)

    data_list = []
    for i in tqdm(range(len(drugs)), desc='Building dataset'):
        smi = drugs[i]
        if smi not in graph_cache:
            continue

        g = graph_cache[smi]
        graph = DATA.Data(
            x=torch.from_numpy(g['x']),
            edge_index=torch.from_numpy(g['edge_index']),
            edge_attr=torch.from_numpy(g['edge_attr']),
            y=torch.tensor([affinities[i]], dtype=torch.float),
        )

        # Protein features (physicochemical 3D or ESM-2 1D)
        prot = proteins[i]
        if prot.ndim == 1:
            # ESM-2 embedding: (esm2_dim,) → store as 1D
            graph.target = torch.from_numpy(prot).unsqueeze(0).float()
        else:
            # Physicochemical: (max_len, 8) → store as 3D
            graph.target = torch.from_numpy(prot).unsqueeze(0).float()

        graph.target_seq = padded_tokens[i].unsqueeze(0)
        graph.__setitem__('c_size', torch.tensor([g['num_atoms']], dtype=torch.long))

        # Morgan fingerprint (supplementary drug features)
        if morgan_fp_dim > 0:
            fp = compute_morgan_fp(smi, n_bits=morgan_fp_dim)
            graph.__setitem__('morgan_fp', torch.from_numpy(fp).unsqueeze(0))

        # Molecular descriptor regression targets
        if include_descriptors:
            desc = compute_mol_descriptors(smi)
            if desc_stats is not None:
                desc = normalize_descriptors(desc, desc_stats[0], desc_stats[1])
            graph.__setitem__('mol_descriptors', torch.from_numpy(desc).unsqueeze(0))

        # Functional group multi-label targets
        if include_funcgroups:
            fg = compute_functional_groups(smi)
            graph.__setitem__('func_groups', torch.from_numpy(fg).unsqueeze(0))

        # Optional DTI/MoA binary labels
        if labels is not None:
            if 'dti' in labels:
                graph.__setitem__('dti_label', torch.tensor([labels['dti'][i]], dtype=torch.float))
            if 'moa' in labels:
                graph.__setitem__('moa_label', torch.tensor([labels['moa'][i]], dtype=torch.float))

        data_list.append(graph)

    return MolProteinDataset(data_list=data_list, cache_path=cache_path)
