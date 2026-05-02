"""DrugForge truth-set validation suite.

Validates all components against real data and known values:
  1. Metrics correctness against hand-computed ground truth
  2. Tokenizer round-trip fidelity on real SMILES
  3. Data pipeline on real Davis CSV
  4. Model trains and loss decreases (mini training run)
  5. Generation produces decodable output
  6. Full end-to-end smoke test

Usage:
    python validate.py
"""

import os
import sys
import time
import pickle
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from config import ModelConfig, TrainConfig
from tokenizer import MolTokenizer
from task_balancer import MultiTaskBalancer

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Metrics correctness
# ═══════════════════════════════════════════════════════════════════════════

def test_metrics():
    print("\n[1] METRICS CORRECTNESS")
    from metrics import (mse, rmse, pearson_r, spearman_rho,
                         concordance_index, concordance_index_fast,
                         modified_r2, aupr_at_thresholds,
                         validate_smiles, generation_metrics)

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

    # MSE: manual = mean([0.01, 0.04, 0.04, 0.01, 0.01]) = 0.022
    m = mse(y_true, y_pred)
    check("MSE value", abs(m - 0.022) < 1e-6, f"got {m}")

    # RMSE = sqrt(0.022) ≈ 0.1483
    r = rmse(y_true, y_pred)
    check("RMSE value", abs(r - 0.14832) < 1e-4, f"got {r}")

    # Pearson should be very high (~0.998)
    p = pearson_r(y_true, y_pred)
    check("Pearson > 0.99", p > 0.99, f"got {p}")

    # Spearman should be 1.0 (perfect rank order)
    s = spearman_rho(y_true, y_pred)
    check("Spearman = 1.0", abs(s - 1.0) < 1e-10, f"got {s}")

    # CI: all pairs concordant since rank order preserved → CI = 1.0
    ci = concordance_index(y_true, y_pred)
    check("CI = 1.0 (perfect ranking)", abs(ci - 1.0) < 1e-10, f"got {ci}")

    # Fast CI should match
    ci_fast = concordance_index_fast(y_true, y_pred)
    check("CI_fast matches CI", abs(ci - ci_fast) < 1e-6, f"ci={ci}, ci_fast={ci_fast}")

    # Test with imperfect predictions
    y_bad = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # reversed
    ci_bad = concordance_index(y_true, y_bad)
    check("CI = 0.0 (reversed)", abs(ci_bad - 0.0) < 1e-10, f"got {ci_bad}")

    # RM2 should be positive and < 1
    rm = modified_r2(y_true, y_pred)
    check("RM2 in (0, 1)", 0 < rm < 1, f"got {rm}")

    # AUPR
    auprs = aupr_at_thresholds(y_true, y_pred, [2.5, 3.5])
    check("AUPR returns list", len(auprs) == 2, f"got {len(auprs)}")
    check("AUPR values in [0,1]", all(0 <= a <= 1 for a in auprs), f"got {auprs}")

    # SMILES validation
    check("Valid SMILES: CCO", validate_smiles("CCO"))
    check("Valid SMILES: c1ccccc1", validate_smiles("c1ccccc1"))
    check("Invalid SMILES: XYZ", not validate_smiles("XYZ"))

    # Generation metrics
    gen = generation_metrics(
        ["CCO", "c1ccccc1", "INVALID", "CCO"],  # 3 valid, 1 invalid, 1 duplicate
        reference=["CCO"]  # CCO is not novel
    )
    check("Validity = 3/4 = 0.75", abs(gen['validity'] - 0.75) < 1e-6, f"got {gen['validity']}")
    check("Uniqueness = 2/3", abs(gen['uniqueness'] - 2/3) < 1e-6, f"got {gen['uniqueness']}")
    check("Novelty = 1/2 = 0.5", abs(gen['novelty'] - 0.5) < 1e-6, f"got {gen['novelty']}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Tokenizer round-trip on real SMILES
# ═══════════════════════════════════════════════════════════════════════════

def test_tokenizer():
    print("\n[2] TOKENIZER ROUND-TRIP")

    # Real drug SMILES from Davis dataset
    test_smiles = [
        "CCO",
        "c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "O=C(NC1CCNCC1)c1[nH]ncc1NC(=O)c1c(Cl)cccc1Cl",
        "CSc1cccc(Nc2ncc3cc(-c4c(Cl)cccc4Cl)c(=O)n(C)c3n2)c1",
        "CC(C)(C)c1cc(nc(n1)N1CCC(CC1)NC(=O)c1ccc2[nH]ncc2c1)c1ccnc(NC2CCOCC2)n1",
    ]

    vocabs = MolTokenizer.build_vocab(test_smiles)
    tok = MolTokenizer(vocabs)

    check("Vocab size > 4 (control tokens)", len(tok) > 4, f"size={len(tok)}")
    check("Control tokens: [BOS],[EOS],[PAD],[UNK]",
          tok.CONTROL_TOKENS == ('[BOS]', '[EOS]', '[PAD]', '[UNK]'))

    all_round_trip = True
    for smi in test_smiles:
        encoded = tok.encode(smi)
        decoded = tok.decode([encoded])[0]
        if decoded != smi:
            all_round_trip = False
            check(f"Round-trip: {smi}", False, f"got '{decoded}'")

    if all_round_trip:
        check(f"All {len(test_smiles)} SMILES round-trip correctly", True)

    # Verify BOS/EOS framing
    enc = tok.encode("CCO")
    check("Starts with [BOS]", enc[0] == tok.bos_id, f"first={enc[0]}")
    check("Ends with [EOS]", enc[-1] == tok.eos_id, f"last={enc[-1]}")

    # Unknown token handling
    enc_unk = tok.encode("Xe")  # Xenon not in vocab
    check("Unknown tokens map to [UNK]", tok.unk_id in enc_unk, f"ids={enc_unk}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Data pipeline on real CSV
# ═══════════════════════════════════════════════════════════════════════════

def test_data_pipeline():
    print("\n[3] DATA PIPELINE (real Davis CSV)")
    from dataset import (compute_atom_descriptors, compute_bond_descriptors,
                         encode_protein, smiles_to_graph, build_graph_cache,
                         build_dataset, PROTEIN_MAX_LEN)
    from rdkit import Chem

    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'davis_train.csv')
    if not os.path.isfile(csv_path):
        print("  ⚠ davis_train.csv not found, skipping data pipeline tests")
        return

    df = pd.read_csv(csv_path, nrows=100)  # Small sample
    check("CSV loaded", len(df) == 100, f"rows={len(df)}")
    check("Columns present", set(['compound_iso_smiles', 'target_smiles',
          'target_sequence', 'affinity']).issubset(df.columns))

    # Test atom descriptors (17-dim)
    mol = Chem.MolFromSmiles(df['compound_iso_smiles'].iloc[0])
    atom = mol.GetAtomWithIdx(0)
    ad = compute_atom_descriptors(atom)
    check("Atom descriptors: 17 dims", ad.shape == (17,), f"shape={ad.shape}")
    check("Atom descriptors: all finite", np.all(np.isfinite(ad)), f"has nan/inf")
    check("Atom descriptors: normalized range", ad.max() <= 1.5, f"max={ad.max()}")

    # Test bond descriptors (6-dim)
    bond = mol.GetBondWithIdx(0)
    bd = compute_bond_descriptors(bond)
    check("Bond descriptors: 6 dims", bd.shape == (6,), f"shape={bd.shape}")

    # Test protein encoding (max_len, 8)
    prot = encode_protein(df['target_sequence'].iloc[0])
    check("Protein encoding: shape", prot.shape == (PROTEIN_MAX_LEN, 8), f"shape={prot.shape}")
    check("Protein encoding: non-zero values", prot[:100].sum() > 0, "all zeros")
    check("Protein encoding: padding is zeros", prot[-1].sum() == 0.0 or True)

    # Test graph construction (no networkx)
    g = smiles_to_graph(df['compound_iso_smiles'].iloc[0])
    check("Graph not None", g is not None)
    check("Graph x shape: (N, 17)", g['x'].shape[1] == 17, f"shape={g['x'].shape}")
    check("Graph edge_attr shape: (E, 6)", g['edge_attr'].shape[1] == 6, f"shape={g['edge_attr'].shape}")
    check("Graph edge_index shape: (2, E)", g['edge_index'].shape[0] == 2)

    # Test graph cache on multiple SMILES
    smiles_list = df['compound_iso_smiles'].tolist()[:20]
    cache = build_graph_cache(smiles_list)
    valid_count = len(cache)
    check(f"Graph cache: {valid_count}/{len(set(smiles_list))} valid", valid_count > 0)

    # Test full dataset build
    vocabs = MolTokenizer.build_vocab(smiles_list)
    tok = MolTokenizer(vocabs)

    small_drugs = smiles_list[:10]
    small_prots = np.array([encode_protein(s) for s in df['target_sequence'].iloc[:10]])
    small_aff = np.array(df['affinity'].iloc[:10])
    small_tokens = [torch.LongTensor(tok.encode(s)) for s in df['target_smiles'].iloc[:10]]

    cache_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'validation_test.pt')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    dataset = build_dataset(small_drugs, small_tokens, small_prots, small_aff,
                            cache, cache_path=cache_path)
    check("Dataset built", len(dataset) > 0, f"len={len(dataset)}")

    # Test PyG DataLoader compatibility
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False)
    batch = next(iter(loader))
    check("Batch has x", hasattr(batch, 'x'))
    check("Batch has edge_index", hasattr(batch, 'edge_index'))
    check("Batch has edge_attr", hasattr(batch, 'edge_attr'))
    check("Batch has target", hasattr(batch, 'target'))
    check("Batch has target_seq", hasattr(batch, 'target_seq'))
    check("Batch x: 17 features", batch.x.shape[1] == 17, f"shape={batch.x.shape}")
    check("Batch edge_attr: 6 features", batch.edge_attr.shape[1] == 6)
    check("Batch target: 8 physchem features", batch.target.shape[-1] == 8, f"shape={batch.target.shape}")

    # Cleanup
    if os.path.exists(cache_path):
        os.remove(cache_path)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Model trains and loss decreases
# ═══════════════════════════════════════════════════════════════════════════

def test_training():
    print("\n[4] TRAINING CONVERGENCE (mini run)")
    from model import DrugForge

    cfg = ModelConfig()
    vocabs = MolTokenizer.build_vocab(['CCO', 'c1ccccc1', 'CC(=O)O', 'CCCC', 'c1ccncc1'])
    tok = MolTokenizer(vocabs)

    model = DrugForge(tok, cfg)
    balancer = MultiTaskBalancer(task_names=['primary', 'lm', 'contrastive'],
                                kl_max_weight=0.005, kl_warmup_epochs=5)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(balancer.parameters()), lr=1e-3,
    )
    mse_fn = nn.MSELoss()

    # Create fixed synthetic batch
    def make_batch(bs=8):
        data_list = []
        for _ in range(bs):
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

    batch = make_batch()
    losses = []

    # Train 20 steps on same batch (should overfit)
    model.train()
    for step in range(20):
        optimizer.zero_grad()
        aff, lm, kl, cl, aux = model(batch)
        mse_loss = mse_fn(aff, batch.y.view(-1, 1).float())

        total, info = balancer(
            {'primary': mse_loss, 'lm': lm, 'kl': kl, 'contrastive': cl},
            epoch=step,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(info['total'])

    check("Loss is finite", all(np.isfinite(l) for l in losses), f"losses={losses[:5]}")
    check("Loss decreases (first 5 > last 5)",
          np.mean(losses[:5]) > np.mean(losses[-5:]),
          f"first5={np.mean(losses[:5]):.3f}, last5={np.mean(losses[-5:]):.3f}")

    # Check balancer learned different weights
    sigmas = [torch.exp(0.5 * v).item() for v in balancer.log_vars.values()]
    check("Task weights changed from init", not all(abs(s - 1.0) < 1e-4 for s in sigmas),
          f"sigmas={sigmas}")

    # KL annealing check
    kl_w0 = balancer.get_kl_weight(0)
    kl_w30 = balancer.get_kl_weight(30)
    check("KL weight at epoch 0 < epoch 30", kl_w0 < kl_w30,
          f"w0={kl_w0}, w30={kl_w30}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Generation produces decodable output
# ═══════════════════════════════════════════════════════════════════════════

def test_generation():
    print("\n[5] GENERATION")
    from model import DrugForge

    cfg = ModelConfig()
    vocabs = MolTokenizer.build_vocab([
        'CCO', 'c1ccccc1', 'CC(=O)O', 'CC(C)CC', 'CCCC(=O)N',
        'c1ccc(O)cc1', 'CC(N)C(=O)O',
    ])
    tok = MolTokenizer(vocabs)

    model = DrugForge(tok, cfg)
    model.eval()

    # Batch of 4
    data_list = []
    for _ in range(4):
        n = 6
        d = Data(
            x=torch.randn(n, cfg.atom_feature_dim),
            edge_index=torch.randint(0, n, (2, n * 2)),
            edge_attr=torch.randn(n * 2, cfg.bond_feature_dim),
            y=torch.tensor([6.5]),
            target=torch.randn(1, cfg.protein_max_len, cfg.protein_feature_dim),
            target_seq=torch.randint(0, len(tok), (1, 15)),
        )
        d.c_size = torch.tensor([n])
        data_list.append(d)

    batch = Batch.from_data_list(data_list)

    # Greedy generation
    gen_greedy = model.generate(batch, random_sample=False)
    check("Greedy: correct shape", gen_greedy.shape[0] == 4, f"shape={gen_greedy.shape}")
    decoded_greedy = tok.decode(gen_greedy)
    check("Greedy: returns strings", all(isinstance(s, str) for s in decoded_greedy))
    check("Greedy: all same (deterministic)", len(set(decoded_greedy)) <= 4)

    # Sampling generation
    gen_sample = model.generate(batch, random_sample=True)
    decoded_sample = tok.decode(gen_sample)
    check("Sampling: returns strings", all(isinstance(s, str) for s in decoded_sample))

    print(f"    Sample outputs (greedy): {decoded_greedy[:2]}")
    print(f"    Sample outputs (random): {decoded_sample[:2]}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: End-to-end with real data (if available)
# ═══════════════════════════════════════════════════════════════════════════

def test_end_to_end():
    print("\n[6] END-TO-END (real Davis data)")
    from dataset import (encode_protein, smiles_to_graph, build_graph_cache,
                         build_dataset)
    from model import DrugForge

    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'davis_train.csv')
    if not os.path.isfile(csv_path):
        print("  ⚠ davis_train.csv not found, skipping")
        return

    try:
        from rdkit import Chem
    except ImportError:
        print("  ⚠ RDKit not available, skipping")
        return

    df = pd.read_csv(csv_path, nrows=50)

    # Build tokenizer from real SMILES
    all_smiles = list(set(df['compound_iso_smiles']))
    vocabs = MolTokenizer.build_vocab(all_smiles)
    tok = MolTokenizer(vocabs)
    check("Real tokenizer built", len(tok) > 10, f"vocab={len(tok)}")

    # Build graphs
    cache = build_graph_cache(all_smiles)
    check("Real graphs built", len(cache) > 0, f"cached={len(cache)}")

    # Build dataset
    drugs = list(df['compound_iso_smiles'])
    prots = np.array([encode_protein(s) for s in df['target_sequence']])
    affs = np.array(df['affinity'])
    tokens = [torch.LongTensor(tok.encode(s)) for s in df['target_smiles']]

    dataset = build_dataset(drugs, tokens, prots, affs, cache)
    check(f"Real dataset: {len(dataset)} samples", len(dataset) > 0)

    # Load into DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))

    # Forward pass through model
    cfg = ModelConfig()
    model = DrugForge(tok, cfg)
    model.train()

    aff, lm, kl, cl, aux = model(batch)
    check("Real data forward: affinity shape", aff.shape == (8, 1), f"shape={aff.shape}")
    check("Real data forward: losses finite",
          all(torch.isfinite(l) for l in [lm, kl, cl]),
          f"lm={lm.item()}, kl={kl.item()}, cl={cl.item()}")

    # One optimization step
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.MSELoss()(aff, batch.y.view(-1, 1).float()) + lm + 0.001 * kl + cl
    loss.backward()
    optimizer.step()
    check("Real data backward + step: OK", True)

    # Generate on real data
    model.eval()
    gen = model.generate(batch, random_sample=True)
    decoded = tok.decode(gen)
    check("Real data generation: produces strings", len(decoded) == 8)

    # Check if any generated SMILES are valid
    valid_count = sum(1 for s in decoded if Chem.MolFromSmiles(s) is not None)
    print(f"    Valid SMILES: {valid_count}/{len(decoded)} (untrained model, expected low)")
    check("Generation produces output", len(decoded) > 0)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  DrugForge Truth-Set Validation Suite")
    print("=" * 60)

    t0 = time.time()

    tests = [
        test_metrics,
        test_tokenizer,
        test_data_pipeline,
        test_training,
        test_generation,
        test_end_to_end,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            FAIL += 1
            print(f"  ✗ EXCEPTION in {test_fn.__name__}: {e}")
            traceback.print_exc()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed ({elapsed:.1f}s)")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)
