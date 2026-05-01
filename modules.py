"""DrugForge neural network modules — all original components.

  1. GINEncoder          — GIN with edge features + virtual node
  2. ProteinTransformer  — Self-attention on continuous physicochemical features
  3. BidirectionalCrossAttention — Drug↔Protein cross-modal attention
  4. CrossAttentionVAE   — VAE conditioned via cross-attention (softplus variance)
  5. MoleculeDecoder     — Transformer decoder for molecular generation
  6. AffinityHead        — MLP on cross-attended features
  7. ContrastiveLoss     — InfoNCE for drug-protein alignment
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

from config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════════
# Learnable Positional Encoding
# ═══════════════════════════════════════════════════════════════════════════

class LearnablePositionalEncoding(nn.Module):
    """Learned positional embeddings (no sinusoidal formula)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return self.dropout(x + self.pos_embed(positions))


# ═══════════════════════════════════════════════════════════════════════════
# 1. GIN Encoder with Edge Features + Virtual Node
# ═══════════════════════════════════════════════════════════════════════════

class GINEncoder(nn.Module):
    """Graph Isomorphism Network with edge features and virtual node.

    More expressive than GCN for graph-level tasks (Xu et al., 2019).
    Virtual node provides global context aggregation (Li et al., 2020).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_layers = cfg.gin_layers
        h = cfg.gin_hidden_dim

        self.atom_proj = nn.Linear(cfg.atom_feature_dim, h)
        self.bond_proj = nn.Linear(cfg.bond_feature_dim, h)

        self.virtual_node_emb = nn.Embedding(1, h)
        nn.init.zeros_(self.virtual_node_emb.weight)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.virtual_mlps = nn.ModuleList()

        for _ in range(cfg.gin_layers):
            mlp = nn.Sequential(
                nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
                nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            )
            self.convs.append(GINEConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(h))
            self.virtual_mlps.append(
                nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            )

        self.dropout = nn.Dropout(cfg.gin_dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        batch_size = batch.max().item() + 1
        h = self.atom_proj(x)
        edge_h = self.bond_proj(edge_attr)
        vn = self.virtual_node_emb.weight.expand(batch_size, -1)

        for i in range(self.num_layers):
            h = h + vn[batch]
            h = self.convs[i](h, edge_index, edge_h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = self.dropout(h)

            if i < self.num_layers - 1:
                vn_agg = global_add_pool(h, batch)
                vn = vn + self.virtual_mlps[i](vn_agg)

        graph_emb = global_add_pool(h, batch) + vn
        return h, graph_emb


# ═══════════════════════════════════════════════════════════════════════════
# 2. Protein Transformer Encoder (continuous physicochemical input)
# ═══════════════════════════════════════════════════════════════════════════

class ProteinTransformer(nn.Module):
    """Transformer encoder for protein sequences with continuous features.

    Takes 8-dim physicochemical descriptors per residue (not discrete tokens).
    Uses a learnable CLS token for global representation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.protein_embed_dim

        # Linear projection from continuous features (not nn.Embedding)
        self.input_proj = nn.Linear(cfg.protein_feature_dim, d)
        self.pos_enc = LearnablePositionalEncoding(d, max_len=cfg.protein_max_len + 1,
                                                   dropout=cfg.protein_dropout)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.protein_heads,
            dim_feedforward=cfg.protein_ff_dim,
            dropout=cfg.protein_dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.protein_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, protein_feats: torch.Tensor):
        """
        Args:
            protein_feats: (batch, max_len, 8) continuous physicochemical features

        Returns:
            residue_emb: (batch, seq_len+1, dim) — residues + CLS
            global_emb: (batch, dim) — CLS token
            padding_mask: (batch, seq_len+1) — True for padded positions
        """
        B = protein_feats.size(0)

        # Project continuous features
        h = self.input_proj(protein_feats)  # (B, L, D)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (B, L+1, D)
        h = self.pos_enc(h)

        # Padding mask: all-zero rows in original features are padding
        pad_mask = (protein_feats.abs().sum(dim=-1) == 0)  # (B, L)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=protein_feats.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)  # (B, L+1)

        h = self.transformer(h, src_key_padding_mask=full_mask)
        h = self.norm(h)

        return h, h[:, 0], full_mask


# ═══════════════════════════════════════════════════════════════════════════
# 3. Bidirectional Cross-Attention
# ═══════════════════════════════════════════════════════════════════════════

class CrossAttentionLayer(nn.Module):
    """Cross-attention: queries from one modality, K/V from another."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, kv, kv_mask=None):
        residual = query
        out, _ = self.cross_attn(query, kv, kv, key_padding_mask=kv_mask)
        out = self.norm1(residual + out)
        out = self.norm2(out + self.ff(out))
        return out


class BidirectionalCrossAttention(nn.Module):
    """Multi-layer bidirectional drug↔protein cross-attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.gin_hidden_dim
        self.layers = cfg.cross_attn_layers
        self.drug_to_protein = nn.ModuleList([
            CrossAttentionLayer(d, cfg.cross_attn_heads, cfg.cross_attn_dropout)
            for _ in range(cfg.cross_attn_layers)
        ])
        self.protein_to_drug = nn.ModuleList([
            CrossAttentionLayer(d, cfg.cross_attn_heads, cfg.cross_attn_dropout)
            for _ in range(cfg.cross_attn_layers)
        ])

    def forward(self, drug_emb, drug_mask, protein_emb, protein_mask):
        d, p = drug_emb, protein_emb
        for i in range(self.layers):
            d = self.drug_to_protein[i](d, p, kv_mask=protein_mask)
            p = self.protein_to_drug[i](p, d, kv_mask=drug_mask)
        return d, p


# ═══════════════════════════════════════════════════════════════════════════
# 4. Cross-Attention Conditioned VAE (softplus variance)
# ═══════════════════════════════════════════════════════════════════════════

class CrossAttentionVAE(nn.Module):
    """VAE with cross-attention protein conditioning and softplus variance.

    Protein features serve as K/V in cross-attention while drug latent
    serves as Q, enabling selective attention-weighted conditioning.
    Uses softplus for variance to ensure smooth, always-positive values.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.latent_dim

        self.mean_net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.logvar_net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

        # Cross-attention conditioning
        self.conditioning_attn = nn.MultiheadAttention(
            d, cfg.vae_cross_attn_heads, dropout=0.1, batch_first=True,
        )
        self.cond_norm = nn.LayerNorm(d)

        # Affinity embedding
        self.affinity_proj = nn.Sequential(
            nn.Linear(1, d), nn.GELU(), nn.Linear(d, d),
        )

    def forward(self, drug_seq, protein_emb, protein_mask, affinity):
        """
        Returns:
            z: (B, max_atoms, D) conditioned latent
            kl_loss: scalar KL divergence
        """
        mu = self.mean_net(drug_seq)
        raw_logvar = self.logvar_net(drug_seq)

        # Softplus-based variance (always positive, smooth gradient)
        # log_var = log(softplus(raw)) which is bounded and smooth
        var = F.softplus(raw_logvar)
        log_var = torch.log(var + 1e-8)

        # Reparameterize
        std = torch.sqrt(var)
        z = mu + std * torch.randn_like(std)

        # KL divergence: -0.5 * mean(1 + log_var - mu^2 - var)
        kl_loss = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - var)

        # Cross-attention conditioning on protein
        z_cond, _ = self.conditioning_attn(
            z, protein_emb, protein_emb, key_padding_mask=protein_mask,
        )
        z = self.cond_norm(z + z_cond)

        # Affinity signal
        aff_emb = self.affinity_proj(affinity.unsqueeze(-1))
        z = z + aff_emb.unsqueeze(1)

        return z, kl_loss


# ═══════════════════════════════════════════════════════════════════════════
# 5. Molecule Decoder
# ═══════════════════════════════════════════════════════════════════════════

class MoleculeDecoder(nn.Module):
    """Transformer decoder for autoregressive molecular string generation."""

    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        d = cfg.decoder_dim

        self.word_embed = nn.Embedding(vocab_size, d)
        self.pos_enc = LearnablePositionalEncoding(d, max_len=cfg.max_gen_len + 10,
                                                   dropout=cfg.decoder_dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.decoder_heads,
            dim_feedforward=cfg.decoder_ff_dim,
            dropout=cfg.decoder_dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder_layers)
        self.norm = nn.LayerNorm(d)

        self.head = nn.Sequential(
            nn.Linear(d, d), nn.PReLU(), nn.LayerNorm(d),
            nn.Linear(d, vocab_size),
        )
        nn.init.zeros_(self.head[3].bias)

    def forward(self, tgt_tokens, memory, memory_mask=None):
        T = tgt_tokens.size(1)
        tgt = self.pos_enc(self.word_embed(tgt_tokens))
        causal_mask = torch.triu(
            torch.ones(T, T, device=tgt.device, dtype=torch.bool), diagonal=1,
        )
        out = self.transformer(
            tgt, memory, tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask,
        )
        return self.head(self.norm(out))

    @torch.no_grad()
    def generate(self, memory, memory_mask, bos_id, eos_id, pad_id,
                 max_len=128, random_sample=False):
        B, device = memory.size(0), memory.device
        tokens = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        tokens[:, 0] = bos_id
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(1, max_len):
            tgt = self.pos_enc(self.word_embed(tokens[:, :t]))
            causal = torch.triu(
                torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1,
            )
            out = self.transformer(
                tgt, memory, tgt_mask=causal,
                memory_key_padding_mask=memory_mask,
            )
            logits = self.head(self.norm(out[:, -1]))

            if random_sample:
                next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
            else:
                next_tok = logits.argmax(dim=-1)

            tokens[:, t] = next_tok
            finished |= (next_tok == eos_id)
            if finished.all():
                break

        return tokens[:, 1:]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Affinity Prediction Head
# ═══════════════════════════════════════════════════════════════════════════

class AffinityHead(nn.Module):
    """MLP from cross-attended drug + protein globals to affinity."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.gin_hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(d * 2, cfg.affinity_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.affinity_dropout),
            nn.Linear(cfg.affinity_hidden, cfg.affinity_hidden // 2),
            nn.ReLU(),
            nn.Dropout(cfg.affinity_dropout),
            nn.Linear(cfg.affinity_hidden // 2, 1),
        )

    def forward(self, drug_global, protein_global):
        return self.layers(torch.cat([drug_global, protein_global], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════
# 7. Contrastive Loss
# ═══════════════════════════════════════════════════════════════════════════

class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for drug-protein alignment."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, drug_emb, protein_emb):
        drug_emb = F.normalize(drug_emb, dim=-1)
        protein_emb = F.normalize(protein_emb, dim=-1)
        logits = drug_emb @ protein_emb.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ═══════════════════════════════════════════════════════════════════════════
# 8. Molecular Descriptor Prediction Head (auxiliary regression)
# ═══════════════════════════════════════════════════════════════════════════

class DescriptorHead(nn.Module):
    """Predict RDKit molecular descriptors from drug graph representation.

    Auxiliary self-supervised task: the drug encoder must learn representations
    rich enough to reconstruct molecular properties (LogP, TPSA, MolWt, etc.).
    """

    def __init__(self, input_dim: int, num_descriptors: int = 120, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_descriptors),
        )

    def forward(self, drug_global):
        """drug_global: (B, D) → predicted descriptors: (B, num_desc)"""
        return self.net(drug_global)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Functional Group Prediction Head (auxiliary multi-label classification)
# ═══════════════════════════════════════════════════════════════════════════

class FuncGroupHead(nn.Module):
    """Predict functional group presence from drug graph representation.

    Multi-label binary classification: which of ~85 functional groups
    are present in the molecule.
    """

    def __init__(self, input_dim: int, num_groups: int = 85, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_groups),
        )

    def forward(self, drug_global):
        """drug_global: (B, D) → logits: (B, num_groups)"""
        return self.net(drug_global)


# ═══════════════════════════════════════════════════════════════════════════
# 10. DTI / MoA Classification Head
# ═══════════════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """Binary classification head for DTI (interaction) or MoA (mechanism).

    Takes concatenated drug + protein global features and predicts
    a binary outcome (interacting/non-interacting or activator/inhibitor).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, drug_global, protein_global):
        """Returns logit (B, 1) — use sigmoid for probability."""
        return self.net(torch.cat([drug_global, protein_global], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════
# 11. Morgan Fingerprint Fusion Layer
# ═══════════════════════════════════════════════════════════════════════════

class MorganFPFusion(nn.Module):
    """Fuse GIN graph embedding with Morgan fingerprint features.

    Provides complementary structural information: GIN captures local topology
    while Morgan FP captures circular substructure patterns.
    """

    def __init__(self, gin_dim: int, morgan_dim: int, output_dim: int):
        super().__init__()
        self.morgan_proj = nn.Linear(morgan_dim, gin_dim)
        self.gate = nn.Sequential(
            nn.Linear(gin_dim * 2, gin_dim),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Linear(gin_dim, output_dim)

    def forward(self, gin_emb, morgan_fp):
        """
        Args:
            gin_emb: (B, gin_dim) from GIN encoder
            morgan_fp: (B, morgan_dim) Morgan fingerprint bits
        Returns:
            fused: (B, output_dim)
        """
        morgan_proj = self.morgan_proj(morgan_fp)
        gate = self.gate(torch.cat([gin_emb, morgan_proj], dim=-1))
        fused = gin_emb * gate + morgan_proj * (1 - gate)
        return self.output_proj(fused)


# ═══════════════════════════════════════════════════════════════════════════
# 12. ESM-2 Protein Projection
# ═══════════════════════════════════════════════════════════════════════════

class ESM2Projection(nn.Module):
    """Project pre-computed ESM-2 embeddings to model dimension.

    When using ESM-2 (1280-dim), replaces the ProteinTransformer.
    Adds a CLS-like global token via projection.
    """

    def __init__(self, esm2_dim: int = 1280, output_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(esm2_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, esm2_emb):
        """
        Args:
            esm2_emb: (B, 1, esm2_dim) pre-computed ESM-2 embedding
        Returns:
            protein_emb: (B, 1, output_dim) — treated as single "CLS" token
            global_emb: (B, output_dim)
            mask: (B, 1) all False (no padding)
        """
        emb = esm2_emb.squeeze(1)  # (B, esm2_dim)
        projected = self.proj(emb)  # (B, output_dim)
        protein_emb = projected.unsqueeze(1)  # (B, 1, output_dim)
        mask = torch.zeros(emb.size(0), 1, dtype=torch.bool, device=emb.device)
        return protein_emb, projected, mask
