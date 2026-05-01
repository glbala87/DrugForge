"""DrugForge drug generation script.

Usage:
    python generate.py <dataset> [--device cpu|cuda] [--random-sample]
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from config import ModelConfig
from dataset import MolProteinDataset
from model import DrugForge

RDLogger.DisableLog('rdApp.*')
logger = logging.getLogger(__name__)


def load_model(model_path, tokenizer_path, device='cpu'):
    from tokenizer import MolTokenizer
    json_path = tokenizer_path.replace('.pkl', '.json')
    if os.path.isfile(json_path):
        tokenizer = MolTokenizer.load_json(json_path)
    else:
        logger.warning(f'Loading tokenizer from pickle: {tokenizer_path}')
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)

    model = DrugForge(tokenizer, ModelConfig())
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, tokenizer


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def main():
    parser = argparse.ArgumentParser(description='DrugForge: Generate target-specific drugs')
    parser.add_argument('dataset', type=str, choices=['davis', 'kiba', 'bindingdb'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--random-sample', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    model_path = f'saved_models/drugforge_{args.dataset}.pth'
    tokenizer_path = f'data/{args.dataset}_tokenizer.pkl'

    model, tokenizer = load_model(model_path, tokenizer_path, args.device)

    test_cache = f'data/processed/{args.dataset}_test.pt'
    test_data = MolProteinDataset(cache_path=test_cache)
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    output_dir = Path(f'generated_results/{args.dataset}')
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for data in tqdm(loader, desc='Generating'):
        data = data.to(args.device)
        token_ids = model.generate(data, random_sample=args.random_sample)

        if tokenizer.use_selfies:
            selfies_list = tokenizer.decode(token_ids)
            for sf in selfies_list:
                try:
                    generated.append(tokenizer.selfies_to_smiles(sf))
                except Exception:
                    pass
        else:
            generated.extend(tokenizer.decode(token_ids))

    # Validate, canonicalize, deduplicate
    valid = [c for s in generated if (c := canonicalize(s)) is not None]
    unique = list(set(valid))

    output_path = output_dir / 'generated_smiles.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(unique) + '\n')

    logger.info(f'Generated: {len(generated)} | Valid: {len(valid)} | '
                f'Unique: {len(unique)} | Saved: {output_path}')


if __name__ == '__main__':
    main()
