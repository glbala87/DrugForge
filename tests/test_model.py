"""Tests for DrugForge model forward/backward pass and generation."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch

from config import ModelConfig
from model import DrugForge
from tokenizer import MolTokenizer
from task_balancer import MultiTaskBalancer


@pytest.fixture
def model_and_tokenizer():
    vocabs = MolTokenizer.build_vocab(['CCO', 'c1ccccc1', 'CC(=O)O', 'CCCC', 'c1ccncc1'])
    tok = MolTokenizer(vocabs)
    cfg = ModelConfig()
    model = DrugForge(tok, cfg)
    return model, tok, cfg


@pytest.fixture
def synthetic_batch(model_and_tokenizer):
    _, tok, cfg = model_and_tokenizer
    data_list = []
    for _ in range(4):
        n = np.random.randint(3, 10)
        d = Data(
            x=torch.randn(n, cfg.atom_feature_dim),
            edge_index=torch.randint(0, n, (2, n * 2)),
            edge_attr=torch.randn(n * 2, cfg.bond_feature_dim),
            y=torch.tensor([np.random.uniform(4, 10)]),
            target=torch.randn(1, cfg.protein_max_len, cfg.protein_feature_dim),
            target_seq=torch.randint(0, len(tok), (1, 20)),
        )
        d.c_size = torch.tensor([n])
        data_list.append(d)
    return Batch.from_data_list(data_list)


class TestModelForward:
    def test_forward_output_shapes(self, model_and_tokenizer, synthetic_batch):
        model, _, _ = model_and_tokenizer
        model.train()
        primary_pred, lm_loss, kl_loss, cl_loss, aux_losses = model(synthetic_batch)
        assert primary_pred.shape == (4, 1)
        assert lm_loss.dim() == 0
        assert kl_loss.dim() == 0
        assert cl_loss.dim() == 0

    def test_forward_losses_finite(self, model_and_tokenizer, synthetic_batch):
        model, _, _ = model_and_tokenizer
        model.train()
        primary_pred, lm_loss, kl_loss, cl_loss, _ = model(synthetic_batch)
        assert torch.isfinite(lm_loss)
        assert torch.isfinite(kl_loss)
        assert torch.isfinite(cl_loss)
        assert torch.all(torch.isfinite(primary_pred))

    def test_backward_pass(self, model_and_tokenizer, synthetic_batch):
        model, _, _ = model_and_tokenizer
        model.train()
        aff, lm, kl, cl, _ = model(synthetic_batch)
        loss = nn.MSELoss()(aff, synthetic_batch.y.view(-1, 1).float()) + lm + 0.001 * kl + cl
        loss.backward()
        # Check gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0


class TestModelGeneration:
    def test_greedy_generation(self, model_and_tokenizer, synthetic_batch):
        model, tok, _ = model_and_tokenizer
        model.eval()
        gen = model.generate(synthetic_batch, random_sample=False)
        assert gen.shape[0] == 4
        decoded = tok.decode(gen)
        assert all(isinstance(s, str) for s in decoded)

    def test_sampling_generation(self, model_and_tokenizer, synthetic_batch):
        model, tok, _ = model_and_tokenizer
        model.eval()
        gen = model.generate(synthetic_batch, random_sample=True)
        decoded = tok.decode(gen)
        assert len(decoded) == 4


class TestTrainingConvergence:
    def test_loss_decreases(self, model_and_tokenizer, synthetic_batch):
        model, _, _ = model_and_tokenizer
        balancer = MultiTaskBalancer(
            task_names=['primary', 'lm', 'contrastive'],
            kl_max_weight=0.005, kl_warmup_epochs=5,
        )
        optimizer = optim.AdamW(
            list(model.parameters()) + list(balancer.parameters()), lr=1e-3,
        )
        mse_fn = nn.MSELoss()

        losses = []
        model.train()
        for step in range(15):
            optimizer.zero_grad()
            aff, lm, kl, cl, _ = model(synthetic_batch)
            mse_loss = mse_fn(aff, synthetic_batch.y.view(-1, 1).float())
            total, info = balancer(
                {'primary': mse_loss, 'lm': lm, 'contrastive': cl, 'kl': kl},
                epoch=step,
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(info['total'])

        assert all(np.isfinite(l) for l in losses)
        assert np.mean(losses[:5]) > np.mean(losses[-5:])
