"""DrugForge — Multitask framework for DTA/DTI/MoA prediction and drug generation.

Architecture:
    Drug SMILES → Graph → GIN Encoder (+ optional Morgan FP fusion)
    Protein seq → ProteinTransformer OR ESM-2 Projection
                    ↕ Bidirectional Cross-Attention ↕
    Task heads: Affinity (DTA) | Interaction (DTI) | Mechanism (MoA) | Generation
    Auxiliary heads: Molecular Descriptors | Functional Groups | Contrastive

Supports 3 task modes: 'dta', 'dti', 'moa' (or 'multi' for all)
Plus 3 auxiliary self-supervised tasks always active during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from modules import (
    GINEncoder,
    ProteinTransformer,
    ESM2Projection,
    BidirectionalCrossAttention,
    CrossAttentionVAE,
    MoleculeDecoder,
    AffinityHead,
    ClassificationHead,
    DescriptorHead,
    FuncGroupHead,
    MorganFPFusion,
    ContrastiveLoss,
)


class DrugForge(nn.Module):
    """Main DrugForge model — multi-task drug-target modeling + generation.

    Task modes:
        'dta'   — affinity regression + generation + auxiliary tasks
        'dti'   — binary interaction classification + generation + auxiliary
        'moa'   — mechanism of action classification + generation + auxiliary
        'multi' — all prediction heads active
    """

    def __init__(self, tokenizer, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        cfg = self.cfg

        # --- Drug encoder ---
        self.drug_encoder = GINEncoder(cfg)

        # Optional: Morgan FP fusion for enriched drug representation
        self.use_morgan = cfg.use_morgan_fp
        if cfg.use_morgan_fp:
            self.morgan_fusion = MorganFPFusion(
                cfg.gin_hidden_dim, cfg.morgan_fp_dim, cfg.gin_hidden_dim,
            )

        # --- Protein encoder (choose one) ---
        self.use_esm2 = cfg.use_esm2
        if cfg.use_esm2:
            self.protein_encoder = ESM2Projection(cfg.esm2_dim, cfg.protein_embed_dim)
        else:
            self.protein_encoder = ProteinTransformer(cfg)

        # --- Cross-modal interaction ---
        self.cross_attention = BidirectionalCrossAttention(cfg)

        # --- VAE for generation ---
        self.vae = CrossAttentionVAE(cfg)

        # --- Fusion encoder for decoder memory ---
        self.drug_seg_enc = nn.Parameter(torch.randn(cfg.latent_dim) * 0.02)
        self.prot_seg_enc = nn.Parameter(torch.randn(cfg.latent_dim) * 0.02)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim, nhead=cfg.decoder_heads,
            dim_feedforward=cfg.decoder_ff_dim,
            dropout=cfg.decoder_dropout,
            batch_first=True, norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=2)
        self.fusion_norm = nn.LayerNorm(cfg.latent_dim)

        # --- Decoder ---
        vocab_size = len(tokenizer)
        self.decoder = MoleculeDecoder(cfg, vocab_size)
        self.vocab_size = vocab_size

        # --- Primary prediction heads ---
        self.affinity_head = AffinityHead(cfg)           # DTA regression
        self.dti_head = ClassificationHead(cfg.gin_hidden_dim, cfg.affinity_hidden)  # DTI
        self.moa_head = ClassificationHead(cfg.gin_hidden_dim, cfg.affinity_hidden)  # MoA

        # --- Auxiliary task heads ---
        d = cfg.gin_hidden_dim
        if cfg.use_descriptor_head:
            self.descriptor_head = DescriptorHead(d, cfg.num_mol_descriptors)
        if cfg.use_funcgroup_head:
            self.funcgroup_head = FuncGroupHead(d, cfg.num_functional_groups)

        # --- Contrastive alignment ---
        self.drug_proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.protein_proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.contrastive_loss_fn = ContrastiveLoss()

        # --- Token IDs ---
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------

    def _pad_drug_nodes(self, node_emb, num_nodes):
        B = num_nodes.size(0)
        D = node_emb.size(1)
        max_atoms = int(num_nodes.max().item())
        device = node_emb.device

        padded = torch.zeros(B, max_atoms, D, device=device)
        mask = torch.ones(B, max_atoms, dtype=torch.bool, device=device)

        offset = 0
        for i in range(B):
            n = int(num_nodes[i].item())
            padded[i, :n] = node_emb[offset:offset + n]
            mask[i, :n] = False
            offset += n

        return padded, mask

    def _build_memory(self, vae_out, drug_mask, protein_emb, protein_mask):
        drug_mem = vae_out + self.drug_seg_enc
        prot_mem = protein_emb + self.prot_seg_enc
        memory = torch.cat([drug_mem, prot_mem], dim=1)
        memory_mask = torch.cat([drug_mask, protein_mask], dim=1)
        memory = self.fusion_encoder(memory, src_key_padding_mask=memory_mask)
        memory = self.fusion_norm(memory)
        return memory, memory_mask

    # -------------------------------------------------------------------
    # Forward (training with teacher forcing)
    # -------------------------------------------------------------------

    def forward(self, data):
        """Full forward pass.

        Returns:
            affinity_pred: (B, 1) — DTA/DTI/MoA prediction
            lm_loss: scalar — generation loss
            kl_loss: scalar — VAE KL divergence
            contrastive_loss: scalar — InfoNCE alignment
            aux_losses: dict of auxiliary task losses
        """
        cfg = self.cfg

        # ─── Encode drug graph ───
        node_emb, graph_emb = self.drug_encoder(
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        drug_padded, drug_mask = self._pad_drug_nodes(node_emb, data.c_size)

        # ─── Fuse Morgan FP with graph embedding if available ───
        if self.use_morgan and hasattr(data, 'morgan_fp'):
            morgan_fp = data.morgan_fp  # (B, morgan_dim)
            graph_emb = self.morgan_fusion(graph_emb, morgan_fp)

        # ─── Encode protein ───
        protein_emb, protein_global, protein_mask = self.protein_encoder(data.target)

        # ─── Bidirectional cross-attention ───
        cross_drug, cross_protein = self.cross_attention(
            drug_padded, drug_mask, protein_emb, protein_mask,
        )

        # ─── Global representations for prediction heads ───
        drug_mask_f = (~drug_mask).float().unsqueeze(-1)
        cross_drug_global = (cross_drug * drug_mask_f).sum(1) / drug_mask_f.sum(1).clamp(min=1)
        cross_protein_global = cross_protein[:, 0]

        # ─── Primary prediction (based on task mode) ───
        if cfg.task_mode == 'dti':
            primary_pred = self.dti_head(cross_drug_global, cross_protein_global)
        elif cfg.task_mode == 'moa':
            primary_pred = self.moa_head(cross_drug_global, cross_protein_global)
        else:  # 'dta' or 'multi'
            primary_pred = self.affinity_head(cross_drug_global, cross_protein_global)

        # ─── VAE + decoder for generation ───
        vae_out, kl_loss = self.vae(cross_drug, cross_protein, protein_mask, data.y)
        memory, memory_mask = self._build_memory(
            vae_out, drug_mask, cross_protein, protein_mask,
        )

        targets = data.target_seq
        logits = self.decoder(targets, memory, memory_mask)
        shifted_logits = logits[:, :-1].contiguous().view(-1, self.vocab_size)
        shifted_targets = targets[:, 1:].contiguous().view(-1)
        lm_loss = F.cross_entropy(shifted_logits, shifted_targets, ignore_index=self.pad_id)

        # ─── Contrastive loss ───
        drug_z = self.drug_proj(graph_emb)
        protein_z = self.protein_proj(protein_global)
        contrastive_loss = self.contrastive_loss_fn(drug_z, protein_z)

        # ─── Auxiliary task losses ───
        aux_losses = {}

        if cfg.use_descriptor_head and hasattr(data, 'mol_descriptors'):
            desc_pred = self.descriptor_head(graph_emb)
            desc_target = data.mol_descriptors
            aux_losses['descriptor'] = F.mse_loss(desc_pred, desc_target)

        if cfg.use_funcgroup_head and hasattr(data, 'func_groups'):
            fg_pred = self.funcgroup_head(graph_emb)
            fg_target = data.func_groups
            aux_losses['funcgroup'] = F.binary_cross_entropy_with_logits(fg_pred, fg_target)

        # ─── Additional classification heads (multi mode) ───
        if cfg.task_mode == 'multi':
            if hasattr(data, 'dti_label'):
                dti_pred = self.dti_head(cross_drug_global, cross_protein_global)
                aux_losses['dti'] = F.binary_cross_entropy_with_logits(
                    dti_pred, data.dti_label.view(-1, 1))
            if hasattr(data, 'moa_label'):
                moa_pred = self.moa_head(cross_drug_global, cross_protein_global)
                aux_losses['moa'] = F.binary_cross_entropy_with_logits(
                    moa_pred, data.moa_label.view(-1, 1))

        return primary_pred, lm_loss, kl_loss, contrastive_loss, aux_losses

    # -------------------------------------------------------------------
    # Generation (inference)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, data, random_sample: bool = False):
        self.eval()

        node_emb, graph_emb = self.drug_encoder(
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        drug_padded, drug_mask = self._pad_drug_nodes(node_emb, data.c_size)

        if self.use_morgan and hasattr(data, 'morgan_fp'):
            graph_emb = self.morgan_fusion(graph_emb, data.morgan_fp)

        protein_emb, protein_global, protein_mask = self.protein_encoder(data.target)
        cross_drug, cross_protein = self.cross_attention(
            drug_padded, drug_mask, protein_emb, protein_mask,
        )
        vae_out, _ = self.vae(cross_drug, cross_protein, protein_mask, data.y)
        memory, memory_mask = self._build_memory(
            vae_out, drug_mask, cross_protein, protein_mask,
        )

        return self.decoder.generate(
            memory, memory_mask,
            self.bos_id, self.eos_id, self.pad_id,
            max_len=self.cfg.max_gen_len,
            random_sample=random_sample,
        )

    # -------------------------------------------------------------------
    # Predict (inference — returns affinity/DTI/MoA score)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, data):
        """Run prediction only (no generation). Returns prediction logits."""
        self.eval()

        node_emb, graph_emb = self.drug_encoder(
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        drug_padded, drug_mask = self._pad_drug_nodes(node_emb, data.c_size)

        if self.use_morgan and hasattr(data, 'morgan_fp'):
            graph_emb = self.morgan_fusion(graph_emb, data.morgan_fp)

        protein_emb, protein_global, protein_mask = self.protein_encoder(data.target)
        cross_drug, cross_protein = self.cross_attention(
            drug_padded, drug_mask, protein_emb, protein_mask,
        )

        drug_mask_f = (~drug_mask).float().unsqueeze(-1)
        cross_drug_global = (cross_drug * drug_mask_f).sum(1) / drug_mask_f.sum(1).clamp(min=1)
        cross_protein_global = cross_protein[:, 0]

        cfg = self.cfg
        if cfg.task_mode == 'dti':
            return torch.sigmoid(self.dti_head(cross_drug_global, cross_protein_global))
        elif cfg.task_mode == 'moa':
            return torch.sigmoid(self.moa_head(cross_drug_global, cross_protein_global))
        else:
            return self.affinity_head(cross_drug_global, cross_protein_global)
