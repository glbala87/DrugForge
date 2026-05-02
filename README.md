# DrugForge

A multitask deep learning framework for **drug-target affinity (DTA) prediction**, **drug-target interaction (DTI) classification**, **mechanism of action (MoA) classification**, and **conditional drug molecule generation**.

DrugForge jointly learns to predict how strongly a drug binds to a protein target and to generate novel drug candidates conditioned on a specific protein, binding affinity, and reference molecule.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Execution: Complete Pipeline](#execution-complete-pipeline)
- [Execution: Individual Scripts](#execution-individual-scripts)
- [Python API](#python-api)
- [REST API Server](#rest-api-server)
- [Docker Deployment](#docker-deployment)
- [Task Modes](#task-modes)
- [Supported Datasets](#supported-datasets)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Architecture Overview

```
Drug SMILES -----> Molecular Graph -----> GIN Encoder (+ optional Morgan FP fusion)
                                                    |
                                          Bidirectional Cross-Attention
                                                    |
Protein Sequence -> Physicochemical Features -> Protein Transformer (or ESM-2 Projection)
                                                    |
                            +----------+------------+-----------+-----------+
                            |          |            |           |           |
                        Affinity    DTI          MoA       Generation   Auxiliary
                         Head      Head         Head      (VAE+Decoder)  Heads
                        (DTA)    (binary)     (binary)                  (descriptors,
                                                                        func groups,
                                                                        contrastive)
```

**Key components:**

- **GIN Encoder** -- Graph Isomorphism Network with edge features and virtual node for molecular graph encoding
- **Morgan FP Fusion** -- gated fusion of GIN embeddings with Morgan circular fingerprints for enriched drug representation
- **Protein Transformer** -- self-attention encoder over 8-dimensional physicochemical residue features (hydrophobicity, charge, weight, polarity, H-bond donor/acceptor, aromaticity, van der Waals volume)
- **ESM-2 Projection** -- optional: project pre-trained ESM-2 protein language model embeddings (1280-dim) to model dimension
- **Bidirectional Cross-Attention** -- multi-layer drug-to-protein and protein-to-drug cross-attention for interaction modeling
- **Cross-Attention VAE** -- variational autoencoder conditioned on protein features via cross-attention with softplus variance, affinity-guided latent space
- **Molecule Decoder** -- Transformer decoder for autoregressive SMILES generation from the conditioned latent space
- **Uncertainty-Weighted Multitask Balancer** -- learned task weights (Kendall et al., 2018) with KL beta-annealing

---

## Installation

### System Requirements

- **OS:** Linux, macOS, or Windows (WSL recommended)
- **Python:** 3.10, 3.11, or 3.12
- **RAM:** 8 GB minimum (16 GB recommended for training)
- **GPU:** Optional. NVIDIA GPU with CUDA 11.8+ for accelerated training
- **Disk:** ~2 GB for dependencies, ~1.5 GB for processed datasets

### Tested Dependency Versions

| Package | Version |
|---------|---------|
| Python | 3.13.9 |
| PyTorch | 2.9.1 |
| PyTorch Geometric | 2.7.0 |
| RDKit | 2026.03.1 |
| scikit-learn | 1.7.2 |
| scipy | 1.16.3 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |

### Step 1: Clone the Repository

```bash
git clone https://github.com/glbala87/DrugForge.git
cd DrugForge
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### Step 3: Install PyTorch

Choose the right command for your hardware:

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# NVIDIA GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS) -- use the default pip install
pip install torch
```

### Step 4: Install PyTorch Geometric

```bash
# After PyTorch is installed, install PyG extensions matching your PyTorch + CUDA version
# Example for PyTorch 2.4 + CPU:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install torch-geometric

# For GPU (CUDA 12.1):
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch-geometric
```

See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for other combinations.

### Step 5: Install DrugForge

```bash
# Core installation
pip install -e .
```

### Step 6: Install Optional Extras

```bash
# REST API server (FastAPI + Uvicorn)
pip install -e ".[api]"

# ESM-2 protein language model support
pip install -e ".[esm]"

# Development and testing tools (pytest)
pip install -e ".[dev]"

# Everything at once
pip install -e ".[api,esm,dev]"
```

### Step 7: Verify Installation

```bash
# Quick check -- should print no errors
python -c "from model import DrugForge; from config import ModelConfig; print('DrugForge OK')"

# Run the test suite (54 tests, ~60 seconds)
pytest tests/ -v

# Run the full validation suite (61 checks, ~120 seconds)
python validate.py
```

Expected output:
```
pytest:       54 passed
validate.py:  61 passed, 0 failed
```

---

## Execution: Complete Pipeline

This section walks through the entire workflow from raw data to generated drug molecules.

### Step 1: Prepare Your Data

Place CSV files in the `data/` directory with this exact column format:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `compound_iso_smiles` | string | Drug molecule in SMILES notation | `CC(=O)Oc1ccccc1C(=O)O` |
| `target_smiles` | string | SMILES used for decoder training | same as `compound_iso_smiles` |
| `target_sequence` | string | Full protein amino acid sequence | `MKTAYIAKQRQISFVKSH...` |
| `affinity` | float | Binding affinity (pKd for DTA, 0/1 for DTI) | `7.5` |

File naming convention:
```
data/
  davis_train.csv
  davis_test.csv
```

### Step 2: Preprocess Data

```bash
python create_data.py davis
```

This computes molecular graphs, protein feature matrices, tokenizer vocabulary, Morgan fingerprints, molecular descriptors, and functional group labels. Output:
```
data/davis_tokenizer.json          # Molecular tokenizer (safe JSON format)
data/davis_tokenizer.pkl           # Molecular tokenizer (pickle, legacy)
data/davis_desc_stats.pkl          # Descriptor normalization stats
data/processed/davis_train.pt      # Preprocessed training data
data/processed/davis_test.pt       # Preprocessed test data
```

Options:
```bash
python create_data.py davis --esm2      # Use ESM-2 protein embeddings (requires fair-esm)
python create_data.py davis --no-aux    # Skip auxiliary targets (faster)
python create_data.py                   # Process all datasets (davis, kiba, bindingdb)
```

### Step 3: Train the Model

```bash
# Basic training (500 epochs, batch 32, auto device detection)
python train.py --dataset davis

# GPU training with custom settings
python train.py --dataset davis \
    --device cuda:0 \
    --epochs 500 \
    --batch-size 64 \
    --lr 3e-4 \
    --task-mode dta \
    --eval-every 20 \
    --checkpoint-every 50
```

During training you will see:
```
2026-05-02 17:00:00 [train] INFO: DrugForge | Dataset: davis | Mode: dta | Device: cuda:0
2026-05-02 17:00:00 [train] INFO: Batch: 64 | LR: 0.0003 | Epochs: 500
2026-05-02 17:00:00 [train] INFO: Parameters: 12,456,789
Epoch 1: 100%|████████| 125/125 [00:45<00:00] Pri=1.234 LM=4.567 KL=0.001 Aux=0.890
Epoch 2: 100%|████████| 125/125 [00:44<00:00] Pri=1.100 LM=4.321 KL=0.002 Aux=0.856
...
2026-05-02 17:15:00 [train] INFO: Best model saved!
2026-05-02 17:15:00 [train] INFO:   mse: 0.2341
2026-05-02 17:15:00 [train] INFO:   ci: 0.8812
```

Output files:
```
saved_models/drugforge_davis.pth                  # Best model
saved_models/drugforge_davis_checkpoint.pth        # Latest checkpoint (for resume)
logs/log_davis_<timestamp>.txt                     # Loss history
Affinities/estimated_davis.txt                     # Predicted affinities
Affinities/true_davis.txt                          # Ground truth
```

**Resume after interruption:**
```bash
python train.py --dataset davis --resume saved_models/drugforge_davis_checkpoint.pth
```

**Graceful shutdown:** Press `Ctrl+C` once. DrugForge saves a checkpoint and exits cleanly. Press twice to force-quit.

### Step 4: Evaluate the Model

```bash
python evaluate.py --dataset davis
python evaluate.py --dataset davis --device cuda:0 --batch-size 256
```

Output:
```
2026-05-02 17:30:00 [evaluate] INFO: Evaluation (DTA)
2026-05-02 17:30:05 [evaluate] INFO:   MSE: 0.2341
2026-05-02 17:30:05 [evaluate] INFO:   RMSE: 0.4838
2026-05-02 17:30:05 [evaluate] INFO:   CI: 0.8812
2026-05-02 17:30:05 [evaluate] INFO:   RM2: 0.6534
2026-05-02 17:30:05 [evaluate] INFO:   Pearson: 0.8901
2026-05-02 17:30:05 [evaluate] INFO:   Spearman: 0.8756
```

### Step 5: Generate Drug Molecules

**Batch generation from test set:**
```bash
python generate.py davis --random-sample --device cuda:0
```

Output saved to `generated_results/davis/generated_smiles.txt`.

**Interactive generation for a specific protein target:**
```bash
python infer.py \
    --protein "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK..." \
    --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --affinity 7.5 \
    --n-molecules 50 \
    --output candidates.txt
```

Output:
```
2026-05-02 17:45:00 [infer] INFO: DrugForge loaded | 12,456,789 params | cpu
2026-05-02 17:45:05 [infer] INFO: ============================================================
2026-05-02 17:45:05 [infer] INFO:   RESULTS
2026-05-02 17:45:05 [infer] INFO: ============================================================
2026-05-02 17:45:05 [infer] INFO:   Generated:  50
2026-05-02 17:45:05 [infer] INFO:   Valid:       42 (84.0%)
2026-05-02 17:45:05 [infer] INFO:   Unique:      38 (90.5%)
2026-05-02 17:45:05 [infer] INFO:   Novel:       36 (94.7%)
2026-05-02 17:45:05 [infer] INFO: Novel drug candidates:
2026-05-02 17:45:05 [infer] INFO:     1. CC(=O)c1ccc(O)cc1
2026-05-02 17:45:05 [infer] INFO:     2. c1ccc(-c2nccn2)cc1
...
```

---

## Execution: Individual Scripts

### `create_data.py` -- Data Preparation

```bash
python create_data.py <dataset_name> [--esm2] [--no-aux]
```

| Argument | Description |
|----------|-------------|
| `<dataset_name>` | `davis`, `kiba`, `bindingdb`, or omit for all |
| `--esm2` | Use ESM-2 protein embeddings (requires `fair-esm`) |
| `--no-aux` | Skip auxiliary task targets (faster processing) |

### `train.py` -- Model Training

```bash
python train.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `bindingdb` | `davis`, `kiba`, or `bindingdb` |
| `--task-mode` | `dta` | `dta`, `dti`, `moa`, or `multi` |
| `--device` | auto | `cpu`, `cuda:0`, `cuda:1`, etc. |
| `--epochs` | `500` | Number of training epochs |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `3e-4` | AdamW learning rate |
| `--eval-every` | `20` | Evaluate every N epochs |
| `--checkpoint-every` | `50` | Save checkpoint every N epochs |
| `--data-dir` | `data` | Path to data directory |
| `--output-dir` | `.` | Path for output (models, logs) |
| `--resume` | -- | Path to checkpoint to resume from |

### `evaluate.py` -- Model Evaluation

```bash
python evaluate.py --dataset <name> [--device <dev>] [--batch-size <N>]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `davis`, `kiba`, or `bindingdb` |
| `--device` | `cpu` | `cpu` or `cuda:N` |
| `--batch-size` | `128` | Evaluation batch size |

### `generate.py` -- Batch Drug Generation

```bash
python generate.py <dataset> [--device <dev>] [--random-sample] [--batch-size <N>]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `<dataset>` | required | `davis`, `kiba`, or `bindingdb` |
| `--device` | `cpu` | `cpu` or `cuda:N` |
| `--random-sample` | off | Enable stochastic sampling (more diverse) |
| `--batch-size` | `1` | Generation batch size |

### `infer.py` -- Interactive Inference

```bash
# Generate novel drugs
python infer.py --protein "MKTAY..." --smiles "CCO" --n-molecules 20

# From a FASTA file
python infer.py --fasta target.fasta --smiles "CCO" --n-molecules 100

# Predict affinity for a known pair
python infer.py --mode predict --protein "MKTAY..." --smiles "CC(=O)Oc1ccccc1C(=O)O"

# Screen a compound library
python infer.py --mode screen --protein "MKTAY..." --smiles "CCO" --screen-file compounds.txt --output ranked.txt

# Greedy decoding (less diverse)
python infer.py --protein "MKTAY..." --smiles "CCO" --no-random-sample

# Use a specific model and device
python infer.py --protein "MKTAY..." --smiles "CCO" \
    --model saved_models/drugforge_kiba.pth \
    --tokenizer data/kiba_tokenizer.json \
    --device cuda:0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--protein` | -- | Protein amino acid sequence (or use `--fasta`) |
| `--fasta` | -- | Path to FASTA file (mutually exclusive with `--protein`) |
| `--smiles` | required | Reference drug SMILES (structural seed) |
| `--affinity` | `7.0` | Target binding affinity (pKd scale) |
| `--mode` | `generate` | `generate`, `predict`, or `screen` |
| `--screen-file` | -- | File with SMILES to screen (one per line) |
| `--n-molecules` | `20` | Number of molecules to generate |
| `--no-random-sample` | false | Use greedy decoding |
| `--model` | `saved_models/drugforge_davis.pth` | Path to model checkpoint |
| `--tokenizer` | `data/davis_tokenizer.pkl` | Path to tokenizer (`.json` or `.pkl`) |
| `--device` | `cpu` | `cpu` or `cuda:N` |
| `--output` | -- | Save results to file |

---

## Python API

```python
from infer import DrugForgeInference

# Load a trained model
engine = DrugForgeInference(
    model_path="saved_models/drugforge_davis.pth",
    tokenizer_path="data/davis_tokenizer.json",   # or .pkl
    device="cpu",                                  # or "cuda:0"
)

# --- Generate novel drug molecules ---
results = engine.generate_for_target(
    protein_sequence="MKTAYIAKQRQISFVKSH...",
    reference_smiles="CC(=O)Oc1ccccc1C(=O)O",
    affinity_target=7.0,       # desired pKd
    n_molecules=50,
    random_sample=True,        # stochastic sampling
)

print(f"Generated: {results['stats']['total_generated']}")
print(f"Valid:     {results['stats']['valid_count']} ({results['stats']['validity']:.1%})")
print(f"Unique:    {results['stats']['unique_count']} ({results['stats']['uniqueness']:.1%})")
print(f"Novel:     {results['stats']['novel_count']} ({results['stats']['novelty']:.1%})")

for smi in results['novel'][:10]:
    print(f"  {smi}")

# --- Predict binding affinity ---
pred = engine.predict_interaction(
    protein_sequence="MKTAYIAKQRQISFVKSH...",
    drug_smiles="CC(=O)Oc1ccccc1C(=O)O",
)
print(f"Predicted affinity: {pred['affinity']:.2f}")

# --- Screen multiple compounds ---
ranked = engine.screen_compounds(
    protein_sequence="MKTAYIAKQRQISFVKSH...",
    smiles_list=["CCO", "c1ccccc1", "CC(=O)O", "CC(C)CC"],
)
for r in ranked:
    print(f"  {r['affinity']:.3f}  {r['smiles']}")
```

### Return Values

**`generate_for_target()`** returns:

| Key | Type | Description |
|-----|------|-------------|
| `generated` | `list[str]` | All generated SMILES (may include invalid) |
| `valid` | `list[str]` | RDKit-validated canonical SMILES |
| `unique` | `list[str]` | Deduplicated valid SMILES |
| `novel` | `list[str]` | Unique SMILES different from the reference |
| `stats` | `dict` | Counts and ratios (validity, uniqueness, novelty) |

**`predict_interaction()`** returns (keys depend on task mode):

| Key | Task Mode | Description |
|-----|-----------|-------------|
| `affinity` | `dta`, `multi` | Predicted binding affinity (pKd) |
| `dti_probability` | `dti` | Interaction probability (0-1) |
| `interacts` | `dti` | Boolean: probability > 0.5 |
| `moa_probability` | `moa` | Mechanism probability (0-1) |
| `mechanism` | `moa` | `"activation"` or `"inhibition"` |

---

## REST API Server

### Starting the Server

```bash
# Install API dependencies first
pip install -e ".[api]"

# Start with defaults
python api.py

# Or with uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000

# Configure via environment variables
DRUGFORGE_MODEL=saved_models/drugforge_davis.pth \
DRUGFORGE_TOKENIZER=data/davis_tokenizer.json \
DRUGFORGE_DEVICE=cuda:0 \
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DRUGFORGE_MODEL` | `saved_models/drugforge_davis.pth` | Path to model checkpoint |
| `DRUGFORGE_TOKENIZER` | `data/davis_tokenizer.json` | Path to tokenizer file |
| `DRUGFORGE_DEVICE` | `cpu` | Inference device |

Interactive API docs: `http://localhost:8000/docs` (Swagger UI).

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check and model status |
| `POST` | `/predict` | Predict drug-target affinity or interaction |
| `POST` | `/generate` | Generate novel drug molecules for a target |
| `POST` | `/screen` | Screen multiple compounds against a target |

### Request Examples

**Health check:**
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_loaded": true, "task_mode": "dta"}
```

**Predict affinity:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "protein_sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAP...",
    "smiles": "CC(=O)Oc1ccccc1C(=O)O"
  }'
```
```json
{"affinity": 6.82, "dti_probability": null, "interacts": null, "moa_probability": null, "mechanism": null}
```

**Generate molecules:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "protein_sequence": "MKTAYIAKQRQISFVKSH...",
    "reference_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "affinity_target": 7.5,
    "n_molecules": 20,
    "random_sample": true
  }'
```

**Screen compounds:**
```bash
curl -X POST http://localhost:8000/screen \
  -H "Content-Type: application/json" \
  -d '{
    "protein_sequence": "MKTAYIAKQRQISFVKSH...",
    "smiles_list": ["CCO", "c1ccccc1", "CC(=O)O"]
  }'
```

---

## Docker Deployment

```bash
# Build the image
docker build -t drugforge .

# Run the API server (mount model and data volumes)
docker run -p 8000:8000 \
  -v $(pwd)/saved_models:/app/models \
  -v $(pwd)/data:/app/data \
  -e DRUGFORGE_MODEL=/app/models/drugforge_davis.pth \
  -e DRUGFORGE_TOKENIZER=/app/data/davis_tokenizer.json \
  drugforge

# Run with GPU (requires nvidia-docker)
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/saved_models:/app/models \
  -v $(pwd)/data:/app/data \
  -e DRUGFORGE_DEVICE=cuda:0 \
  drugforge
```

---

## Task Modes

| Mode | Primary Task | Loss Function | Use Case |
|------|-------------|---------------|----------|
| `dta` | Affinity regression | MSE | Binding affinity prediction (pKd scale) |
| `dti` | Interaction classification | Binary Cross-Entropy | Binary binding prediction (interacts / does not) |
| `moa` | Mechanism classification | Binary Cross-Entropy | Activator vs. inhibitor classification |
| `multi` | All heads active | Combined | Joint multi-task training |

All modes additionally train:
- **Molecule generation** via VAE + Transformer decoder
- **Molecular descriptor prediction** (RDKit descriptors, auxiliary regression)
- **Functional group prediction** (~85 fr_* patterns, auxiliary multi-label classification)
- **Contrastive alignment** (InfoNCE loss between drug and protein embeddings)

---

## Supported Datasets

| Dataset | Description | Affinity Type | Scale |
|---------|-------------|---------------|-------|
| **Davis** | Kinase inhibitor binding affinities | Kd (dissociation constant) | pKd (log-transformed) |
| **KIBA** | Kinase Inhibitor BioActivity | Integrated bioactivity score | KIBA score |
| **BindingDB** | Large-scale binding affinity database | Ki/Kd/IC50 | pKd (log-transformed) |

---

## Evaluation Metrics

**Regression (DTA mode):**

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| CI | Concordance Index (pairwise ranking accuracy) |
| rm2 | Modified R-squared |
| Pearson r | Pearson correlation coefficient |
| Spearman rho | Spearman rank correlation |
| AUPR@thresholds | Area Under Precision-Recall Curve at multiple binary thresholds |

**Classification (DTI/MoA mode):**

| Metric | Description |
|--------|-------------|
| AUROC | Area Under ROC Curve |
| AUPR | Area Under Precision-Recall Curve |
| Accuracy | Classification accuracy at threshold 0.5 |
| F1 | F1 score at threshold 0.5 |

**Generation:**

| Metric | Description |
|--------|-------------|
| Validity | Fraction of generated SMILES parseable by RDKit |
| Uniqueness | Fraction of valid SMILES that are unique |
| Novelty | Fraction of unique SMILES not in the reference set |

---

## Configuration Reference

### Model Configuration

Edit `config.py` `ModelConfig` or override via CLI where applicable:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_mode` | `dta` | Task mode (`dta`, `dti`, `moa`, `multi`) |
| `gin_hidden_dim` | `256` | GIN encoder hidden dimension |
| `gin_layers` | `3` | Number of GIN convolution layers |
| `gin_dropout` | `0.1` | GIN dropout rate |
| `use_morgan_fp` | `True` | Fuse Morgan fingerprint with graph embedding |
| `morgan_fp_dim` | `256` | Morgan fingerprint bit length |
| `protein_max_len` | `1200` | Maximum protein sequence length |
| `protein_embed_dim` | `256` | Protein encoder embedding dimension |
| `protein_layers` | `4` | Protein Transformer layers |
| `protein_heads` | `8` | Protein Transformer attention heads |
| `use_esm2` | `False` | Use ESM-2 protein language model |
| `cross_attn_layers` | `2` | Bidirectional cross-attention layers |
| `latent_dim` | `256` | VAE latent dimension |
| `decoder_layers` | `6` | Molecule decoder Transformer layers |
| `decoder_heads` | `8` | Molecule decoder attention heads |
| `max_gen_len` | `128` | Maximum generated SMILES length |
| `use_descriptor_head` | `True` | Enable auxiliary molecular descriptor prediction |
| `use_funcgroup_head` | `True` | Enable auxiliary functional group prediction |
| `use_selfies` | `False` | Use SELFIES representation instead of SMILES |

### Training Configuration

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `batch_size` | `32` | `--batch-size` | Training batch size |
| `learning_rate` | `3e-4` | `--lr` | AdamW learning rate |
| `num_epochs` | `500` | `--epochs` | Total training epochs |
| `eval_every` | `20` | `--eval-every` | Evaluate every N epochs |
| `seed` | `42` | -- | Random seed for reproducibility |
| `grad_clip` | `1.0` | -- | Gradient clipping max norm |
| `kl_warmup_epochs` | `30` | -- | KL divergence annealing warmup epochs |
| `kl_max_weight` | `0.005` | -- | Maximum KL loss weight |

---

## Testing

```bash
# Run the full pytest suite (54 tests, ~60s)
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_model.py -v

# Run the validation suite (61 checks, ~120s)
python validate.py
```

**Test coverage:**

| Test file | What it covers |
|-----------|----------------|
| `test_metrics.py` | MSE, RMSE, CI, Pearson, Spearman, AUPR, generation metrics, classification metrics |
| `test_tokenizer.py` | Round-trip encoding/decoding, BOS/EOS framing, unknown tokens, JSON serialization |
| `test_model.py` | Forward pass shapes, loss finiteness, backward gradients, greedy and sampling generation, training convergence |
| `test_data_pipeline.py` | Atom descriptors (17-dim), bond descriptors (6-dim), protein encoding, graph construction, Morgan FP, molecular descriptors, functional groups |
| `test_infer.py` | Protein sequence validation, SMILES validation, edge cases |

CI runs automatically on push/PR via GitHub Actions (`.github/workflows/ci.yml`), testing against Python 3.10, 3.11, and 3.12.

---

## Project Structure

```
DrugForge/
  config.py            Model and training hyperparameters (dataclasses)
  model.py             DrugForge main model (forward, generate, predict)
  modules.py           Neural network components (GIN, Transformer, VAE, heads)
  tokenizer.py         SMILES/SELFIES tokenizer with JSON serialization
  dataset.py           Data processing (featurization, graph construction, dataset)
  metrics.py           Evaluation metrics (CI, AUPR, generation, classification)
  task_balancer.py     Uncertainty-weighted multitask loss balancer
  train.py             Training script with CLI, checkpointing, resume
  evaluate.py          Model evaluation script
  generate.py          Batch drug generation from test set
  infer.py             Interactive inference (generate, predict, screen)
  api.py               FastAPI REST server
  validate.py          Truth-set validation suite (61 checks)
  create_data.py       Data preparation pipeline
  requirements.txt     Pip dependencies
  pyproject.toml       Package metadata, dependencies, build config
  Dockerfile           Container deployment
  .dockerignore        Docker build exclusions
  .gitignore           Git exclusions
  .github/
    workflows/
      ci.yml           GitHub Actions CI pipeline
  tests/
    conftest.py        pytest path configuration
    test_metrics.py    Metric correctness tests
    test_tokenizer.py  Tokenizer tests
    test_model.py      Model forward/backward/generation tests
    test_data_pipeline.py  Featurization and graph construction tests
    test_infer.py      Input validation tests
  data/                CSV datasets + tokenizers
  saved_models/        Trained model checkpoints
  logs/                Training logs
```

---

## Troubleshooting

**PyTorch Geometric fails to install:**
Install `torch-scatter` and `torch-sparse` first with the correct PyTorch+CUDA version URL from [pyg.org/whl](https://data.pyg.org/whl/), then install `torch-geometric`.

**`RuntimeError: CUDA out of memory`:**
Reduce `--batch-size` (try 16 or 8) or use `--device cpu`.

**`FileNotFoundError: data/processed/davis_train.pt`:**
Run `python create_data.py davis` first to preprocess the data.

**`No tokenizer found for davis`:**
Ensure `data/davis_tokenizer.json` or `data/davis_tokenizer.pkl` exists. Run `python create_data.py davis` to generate it.

**`ModuleNotFoundError: No module named 'torch_geometric'`:**
See [Step 4](#step-4-install-pytorch-geometric) of the installation instructions.

**Validation suite fails on test 4 or 6:**
Ensure you have `data/davis_train.csv` in the `data/` directory. Tests 4 and 6 use real Davis data.

**Generation produces mostly invalid SMILES:**
Train the model for at least 200-300 epochs. An untrained model generates random token sequences.

---

## License

MIT
