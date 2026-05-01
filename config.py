"""DrugForge configuration.

All hyperparameters are defined as dataclasses for clarity and type safety.
Supports: DTA regression, DTI classification, MoA classification, drug generation.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""

    # --- Task mode ---
    # 'dta' = affinity regression + generation
    # 'dti' = binary interaction classification + generation
    # 'moa' = mechanism of action classification + generation
    # 'multi' = all tasks jointly
    task_mode: str = 'dta'

    # --- Drug graph encoder (GIN + virtual node) ---
    atom_feature_dim: int = 17       # continuous atom descriptors
    bond_feature_dim: int = 6        # continuous bond descriptors
    morgan_fp_dim: int = 256         # Morgan fingerprint bits (supplementary drug features)
    use_morgan_fp: bool = True       # concatenate Morgan FP with GIN output
    gin_hidden_dim: int = 256
    gin_layers: int = 3
    gin_dropout: float = 0.1

    # --- Protein encoder ---
    # Option A: Physicochemical Transformer (built-in, lightweight)
    protein_feature_dim: int = 8     # physicochemical properties per residue
    protein_max_len: int = 1200
    protein_embed_dim: int = 256
    protein_layers: int = 4
    protein_heads: int = 8
    protein_ff_dim: int = 512
    protein_dropout: float = 0.1

    # Option B: ESM-2 pre-trained embeddings (requires fair-esm package)
    use_esm2: bool = False           # use ESM-2 protein language model
    esm2_model_name: str = 'esm2_t33_650M_UR50D'  # 650M param model
    esm2_dim: int = 1280            # ESM-2 output dimension
    esm2_repr_layer: int = 33       # which layer to extract representations from

    # --- Bidirectional cross-attention ---
    cross_attn_heads: int = 8
    cross_attn_layers: int = 2
    cross_attn_dropout: float = 0.1

    # --- VAE (cross-attention conditioned) ---
    latent_dim: int = 256
    vae_cross_attn_heads: int = 8

    # --- Molecule decoder (Transformer) ---
    decoder_dim: int = 256
    decoder_heads: int = 8
    decoder_layers: int = 6
    decoder_ff_dim: int = 512
    decoder_dropout: float = 0.1
    max_gen_len: int = 128

    # --- Prediction heads ---
    affinity_hidden: int = 256
    affinity_dropout: float = 0.3

    # --- Auxiliary task heads (inspired by DTIAM multi-task) ---
    use_descriptor_head: bool = True     # predict RDKit molecular descriptors
    num_mol_descriptors: int = 217       # total RDKit descriptors (auto-detected)
    use_funcgroup_head: bool = True      # predict functional group presence
    num_functional_groups: int = 85      # number of fr_* functional group patterns

    # --- Representation format ---
    use_selfies: bool = False


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 500
    eval_every: int = 20
    seed: int = 42
    grad_clip: float = 1.0

    # KL annealing
    kl_warmup_epochs: int = 30
    kl_max_weight: float = 0.005

    # Contrastive loss
    contrastive_temperature: float = 0.07


@dataclass
class DatasetThresholds:
    """Binary classification thresholds for AUPR evaluation per dataset."""
    davis: List[float] = field(default_factory=lambda: [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])
    kiba: List[float] = field(default_factory=lambda: [10.0, 10.5, 11.0, 11.5, 12.0, 12.5])
    bindingdb: List[float] = field(default_factory=lambda: [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])

    def get(self, name: str) -> List[float]:
        return getattr(self, name)
