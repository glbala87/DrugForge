# DrugForge

A multitask deep learning framework for **drug-target affinity (DTA) prediction**, **drug-target interaction (DTI) classification**, **mechanism of action (MoA) classification**, and **conditional drug molecule generation**.

DrugForge jointly learns to predict how strongly a drug binds to a protein target and to generate novel drug candidates conditioned on a specific protein, binding affinity, and reference molecule.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install from Source](#install-from-source)
  - [Optional Extras](#optional-extras)
  - [Verify Installation](#verify-installation)
- [Data Preparation](#data-preparation)
  - [Input CSV Format](#input-csv-format)
  - [Running Data Preparation](#running-data-preparation)
- [Training](#training)
  - [Basic Training](#basic-training)
  - [Training Options](#training-options)
  - [Resume from Checkpoint](#resume-from-checkpoint)
  - [Graceful Shutdown](#graceful-shutdown)
  - [Training Output](#training-output)
- [Evaluation](#evaluation)
- [Drug Generation](#drug-generation)
  - [Batch Generation](#batch-generation)
  - [Interactive Inference](#interactive-inference)
  - [Affinity Prediction](#affinity-prediction)
  - [Compound Screening](#compound-screening)
- [Python API](#python-api)
- [REST API Server](#rest-api-server)
  - [Starting the Server](#starting-the-server)
  - [API Endpoints](#api-endpoints)
  - [Request Examples](#request-examples)
- [Docker Deployment](#docker-deployment)
- [Task Modes](#task-modes)
- [Supported Datasets](#supported-datasets)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration Reference](#configuration-reference)
  - [Model Configuration](#model-configuration)
  - [Training Configuration](#training-configuration)
- [Testing](#testing)
- [Project Structure](#project-structure)
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

### Prerequisites

- **Python** >= 3.10
- **PyTorch** >= 2.0
- **CUDA** (optional, for GPU training)

### Install from Source

```bash
# Clone the repository
git clone <repo-url> DrugForge
cd DrugForge

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# Install core dependencies
pip install -e .
```

If PyTorch Geometric installation fails, install it separately following the [official instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html):

```bash
# Example for PyTorch 2.4 + CPU
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install torch-geometric
```

For GPU support, install PyTorch with CUDA first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Optional Extras

```bash
# REST API server (FastAPI + Uvicorn)
pip install -e ".[api]"

# ESM-2 protein language model support
pip install -e ".[esm]"

# Development and testing tools
pip install -e ".[dev]"

# All extras at once
pip install -e ".[api,esm,dev]"
```

### Verify Installation

```bash
# Run the test suite (54 tests)
pytest tests/ -v

# Run the full validation suite (metrics, tokenizer, data pipeline, training, generation)
python validate.py
```

---

## Data Preparation

### Input CSV Format

Place training and test CSV files in the `data/` directory. Each CSV must contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `compound_iso_smiles` | Drug molecule in SMILES format | `CCO`, `c1ccccc1` |
| `target_smiles` | Target SMILES for decoder training | same as `compound_iso_smiles` or a reference |
| `target_sequence` | Protein amino acid sequence | `MKTAYIAKQRQISFVKSH...` |
| `affinity` | Binding affinity value (pKd for DTA, binary 0/1 for DTI) | `7.5` |

File naming convention:
```
data/
  davis_train.csv
  davis_test.csv
  kiba_train.csv
  kiba_test.csv
  bindingdb_train.csv
  bindingdb_test.csv
```

### Running Data Preparation

```bash
# Process a single dataset
python create_data.py davis

# Process all datasets
python create_data.py

# With ESM-2 protein embeddings (requires fair-esm package)
python create_data.py davis --esm2

# Skip auxiliary task targets (faster processing)
python create_data.py davis --no-aux
```

This will generate:
- `data/<dataset>_tokenizer.pkl` and `data/<dataset>_tokenizer.json` -- molecular tokenizer
- `data/processed/<dataset>_train.pt` -- preprocessed training data
- `data/processed/<dataset>_test.pt` -- preprocessed test data
- `data/<dataset>_desc_stats.pkl` -- descriptor normalization statistics (if auxiliary tasks enabled)

---

## Training

### Basic Training

```bash
python train.py --dataset davis
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `bindingdb` | Dataset name: `davis`, `kiba`, or `bindingdb` |
| `--task-mode` | `dta` | Task mode: `dta`, `dti`, `moa`, or `multi` |
| `--device` | auto | Device: `cpu`, `cuda:0`, `cuda:1`, etc. |
| `--epochs` | `500` | Number of training epochs |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `3e-4` | Learning rate |
| `--eval-every` | `20` | Evaluate every N epochs |
| `--checkpoint-every` | `50` | Save checkpoint every N epochs |
| `--data-dir` | `data` | Path to data directory |
| `--output-dir` | `.` | Path for output (models, logs, affinities) |
| `--resume` | -- | Path to checkpoint file to resume training |

**Examples:**

```bash
# Train on KIBA with DTI mode on GPU
python train.py --dataset kiba --task-mode dti --device cuda:0

# Train with custom hyperparameters
python train.py --dataset davis --epochs 300 --batch-size 64 --lr 1e-4

# Train with multi-task mode (all prediction heads active)
python train.py --dataset bindingdb --task-mode multi --epochs 500

# Custom directories
python train.py --dataset davis --data-dir /path/to/data --output-dir /path/to/output
```

### Resume from Checkpoint

If training is interrupted, resume from the last checkpoint:

```bash
python train.py --dataset davis --resume saved_models/drugforge_davis_checkpoint.pth
```

### Graceful Shutdown

Press `Ctrl+C` once during training. DrugForge will:
1. Finish the current epoch
2. Save a full checkpoint (model, optimizer, balancer state, epoch number)
3. Log the resume command
4. Exit cleanly

Pressing `Ctrl+C` twice forces immediate exit.

### Training Output

```
saved_models/
  drugforge_davis.pth                  # Best model (lowest MSE or highest AUROC)
  drugforge_davis_checkpoint.pth       # Latest periodic checkpoint (for resume)
logs/
  log_davis_<timestamp>.txt            # Per-epoch loss breakdown
Affinities/
  estimated_davis.txt                  # Predicted affinities on test set
  true_davis.txt                       # Ground truth affinities
```

---

## Evaluation

```bash
# Evaluate a trained model
python evaluate.py --dataset davis

# With GPU and larger batch size
python evaluate.py --dataset kiba --device cuda:0 --batch-size 256
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `davis`, `kiba`, or `bindingdb` |
| `--device` | `cpu` | `cpu` or `cuda:N` |
| `--batch-size` | `128` | Evaluation batch size |

The script loads the best model from `saved_models/drugforge_<dataset>.pth` and reports all relevant metrics. If generated SMILES exist at `generated_results/<dataset>/generated_smiles.txt`, it also reports generation metrics (validity, uniqueness, novelty).

---

## Drug Generation

### Batch Generation

Generate molecules for all test set entries:

```bash
# Greedy decoding
python generate.py davis

# Stochastic sampling (more diverse)
python generate.py davis --random-sample

# On GPU
python generate.py kiba --device cuda:0 --random-sample --batch-size 4
```

Output is saved to `generated_results/<dataset>/generated_smiles.txt`.

### Interactive Inference

Generate novel drug candidates for a specific protein target:

```bash
# From a protein sequence
python infer.py \
    --protein "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHFIPFIQRESDHEV..." \
    --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --n-molecules 50 \
    --affinity 7.5

# From a FASTA file
python infer.py \
    --fasta target_protein.fasta \
    --smiles "c1ccccc1" \
    --n-molecules 100

# Greedy decoding (less diverse, more conservative)
python infer.py \
    --protein "MKTAY..." \
    --smiles "CCO" \
    --no-random-sample

# Save results to file
python infer.py \
    --protein "MKTAY..." \
    --smiles "CCO" \
    --output generated_drugs.txt
```

### Affinity Prediction

Predict binding affinity for a specific drug-protein pair:

```bash
python infer.py \
    --mode predict \
    --protein "MKTAYIAKQRQISFVKSH..." \
    --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

### Compound Screening

Rank a list of compounds by predicted affinity to a protein target:

```bash
# Create a file with one SMILES per line
echo -e "CCO\nc1ccccc1\nCC(=O)O\nCCCCO" > compounds.txt

python infer.py \
    --mode screen \
    --protein "MKTAYIAKQRQISFVKSH..." \
    --smiles "CCO" \
    --screen-file compounds.txt \
    --output ranked_compounds.txt
```

### Inference CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--protein` | -- | Protein amino acid sequence (mutually exclusive with `--fasta`) |
| `--fasta` | -- | Path to FASTA file (mutually exclusive with `--protein`) |
| `--smiles` | required | Reference drug SMILES (structural seed) |
| `--affinity` | `7.0` | Target binding affinity on pKd scale |
| `--mode` | `generate` | `generate`, `predict`, or `screen` |
| `--screen-file` | -- | File with SMILES to screen (one per line, for `--mode screen`) |
| `--n-molecules` | `20` | Number of molecules to generate |
| `--no-random-sample` | false | Use greedy decoding instead of stochastic sampling |
| `--model` | `saved_models/drugforge_davis.pth` | Path to trained model checkpoint |
| `--tokenizer` | `data/davis_tokenizer.pkl` | Path to tokenizer file (`.json` or `.pkl`) |
| `--device` | `cpu` | `cpu` or `cuda:N` |
| `--output` | -- | Save results to file |

---

## Python API

```python
from infer import DrugForgeInference

# Load a trained model
engine = DrugForgeInference(
    model_path="saved_models/drugforge_davis.pth",
    tokenizer_path="data/davis_tokenizer.json",  # or .pkl
    device="cpu",   # or "cuda:0"
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

**`generate_for_target()`** returns a dict:
| Key | Type | Description |
|-----|------|-------------|
| `generated` | `list[str]` | All generated SMILES (may include invalid) |
| `valid` | `list[str]` | RDKit-validated canonical SMILES |
| `unique` | `list[str]` | Deduplicated valid SMILES |
| `novel` | `list[str]` | Unique SMILES different from the reference |
| `stats` | `dict` | Counts and ratios (validity, uniqueness, novelty) |

**`predict_interaction()`** returns a dict (keys depend on task mode):
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
# Install API dependencies
pip install -e ".[api]"

# Start with default settings
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

Interactive API docs are available at `http://localhost:8000/docs` (Swagger UI).

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
```json
{
  "novel": ["CC(=O)c1ccc(O)cc1", "..."],
  "valid": ["CC(=O)c1ccc(O)cc1", "CC(=O)Oc1ccccc1C(=O)O", "..."],
  "stats": {"total_generated": 20, "valid_count": 16, "unique_count": 14, "novel_count": 13, "validity": 0.8, "uniqueness": 0.875, "novelty": 0.929}
}
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
```json
{"compounds": [{"smiles": "c1ccccc1", "affinity": 7.2}, {"smiles": "CCO", "affinity": 5.8}, {"smiles": "CC(=O)O", "affinity": 5.1}]}
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
| `dti` | Interaction classification | Binary Cross-Entropy | Binary binding prediction (interacts/does not) |
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
| `use_esm2` | `False` | Use ESM-2 protein language model instead of built-in encoder |
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
| `seed` | `42` | -- | Random seed |
| `grad_clip` | `1.0` | -- | Gradient clipping norm |
| `kl_warmup_epochs` | `30` | -- | KL divergence annealing warmup |
| `kl_max_weight` | `0.005` | -- | Maximum KL loss weight |

---

## Testing

```bash
# Run the full pytest suite (54 tests)
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_model.py -v

# Run the validation suite (metrics, tokenizer, data pipeline, training convergence, generation)
python validate.py
```

The test suite covers:
- **Metrics** -- MSE, RMSE, CI, Pearson, Spearman, AUPR, generation metrics, classification metrics
- **Tokenizer** -- round-trip encoding/decoding, BOS/EOS framing, unknown token handling, JSON serialization
- **Model** -- forward pass shapes, loss finiteness, backward pass gradients, greedy and sampling generation, training convergence
- **Data pipeline** -- atom descriptors (17-dim), bond descriptors (6-dim), protein encoding, graph construction, Morgan FP, molecular descriptors, functional groups
- **Input validation** -- protein sequence validation, SMILES validation, edge cases

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
  validate.py          Truth-set validation suite
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
    conftest.py        pytest configuration
    test_metrics.py    Metric correctness tests
    test_tokenizer.py  Tokenizer round-trip and serialization tests
    test_model.py      Model forward/backward/generation tests
    test_data_pipeline.py  Featurization and graph construction tests
    test_infer.py      Input validation tests
  data/
    davis_train.csv    Davis training data
    davis_test.csv     Davis test data
    ...
  saved_models/        Trained model checkpoints
  logs/                Training logs
```

---

## License

MIT
