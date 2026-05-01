"""DrugForge evaluation script.

Supports DTA regression, DTI/MoA classification modes.

Usage:
    python evaluate.py --dataset davis [--device cpu|cuda]
"""

import argparse
import logging
import os
import pickle

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import ModelConfig, DatasetThresholds
from dataset import MolProteinDataset
from metrics import (
    mse, rmse, concordance_index_fast, modified_r2,
    pearson_r, spearman_rho, aupr_at_thresholds,
    classification_metrics, generation_metrics,
)
from model import DrugForge

logger = logging.getLogger(__name__)


def evaluate_model(model, loader, dataset_name, device, task_mode='dta'):
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for data in tqdm(loader, desc='Evaluation'):
            data = data.to(device)
            pred = model.predict(data)
            all_true.append(data.y.view(-1, 1).cpu())
            all_pred.append(pred.cpu())

    y_true = torch.cat(all_true).numpy().flatten()
    y_pred = torch.cat(all_pred).numpy().flatten()

    if task_mode in ('dti', 'moa'):
        return classification_metrics(y_true, y_pred)
    else:
        thresholds = DatasetThresholds().get(dataset_name)
        return {
            'MSE': mse(y_true, y_pred),
            'RMSE': rmse(y_true, y_pred),
            'CI': concordance_index_fast(y_true, y_pred),
            'RM2': modified_r2(y_true, y_pred),
            'Pearson': pearson_r(y_true, y_pred),
            'Spearman': spearman_rho(y_true, y_pred),
            'AUPR_thresh': aupr_at_thresholds(y_true, y_pred, thresholds),
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate DrugForge')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['davis', 'kiba', 'bindingdb'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)

    model_path = f'saved_models/drugforge_{args.dataset}.pth'
    tok_path = f'data/{args.dataset}_tokenizer.pkl'

    from tokenizer import MolTokenizer
    tok_json = tok_path.replace('.pkl', '.json')
    if os.path.isfile(tok_json):
        tokenizer = MolTokenizer.load_json(tok_json)
    else:
        logger.warning(f'Loading tokenizer from pickle: {tok_path}')
        with open(tok_path, 'rb') as f:
            tokenizer = pickle.load(f)

    cfg = ModelConfig()
    model = DrugForge(tokenizer, cfg)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    task_mode = ckpt.get('task_mode', cfg.task_mode)
    model.load_state_dict(state)
    model.to(device)

    test_cache = f'data/processed/{args.dataset}_test.pt'
    test_data = MolProteinDataset(cache_path=test_cache)
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    logger.info(f'Evaluation ({task_mode.upper()})')
    results = evaluate_model(model, loader, args.dataset, device, task_mode)
    for k, v in results.items():
        if isinstance(v, list):
            logger.info(f'  {k}: {[f"{x:.4f}" for x in v]}')
        else:
            logger.info(f'  {k}: {v:.4f}')

    gen_path = f'generated_results/{args.dataset}/generated_smiles.txt'
    try:
        with open(gen_path) as f:
            generated = [l.strip() for l in f if l.strip()]
        logger.info('Drug Generation metrics:')
        gen = generation_metrics(generated)
        for k, v in gen.items():
            logger.info(f'  {k}: {v:.4f}')
    except FileNotFoundError:
        logger.info(f'No generated SMILES at {gen_path}')


if __name__ == '__main__':
    main()
