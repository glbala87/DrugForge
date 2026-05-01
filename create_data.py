"""Data preparation for DrugForge.

Reads CSV datasets, computes molecular graph descriptors, Morgan FP,
molecular descriptors, functional groups, encodes protein features
(physicochemical or ESM-2), builds tokenizer, and saves datasets.

Usage:
    python create_data.py [dataset_name] [--esm2] [--no-aux]
"""

import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from config import ModelConfig
from dataset import (build_graph_cache, encode_protein, build_dataset,
                     compute_descriptor_stats)
from tokenizer import MolTokenizer

logger = logging.getLogger(__name__)


def prepare_dataset(dataset_name: str, data_dir: str = 'data',
                    use_selfies: bool = False, use_esm2: bool = False,
                    include_aux: bool = True, morgan_fp_dim: int = 256):
    train_csv = os.path.join(data_dir, f'{dataset_name}_train.csv')
    test_csv = os.path.join(data_dir, f'{dataset_name}_test.csv')
    train_cache = os.path.join(data_dir, 'processed', f'{dataset_name}_train.pt')
    test_cache = os.path.join(data_dir, 'processed', f'{dataset_name}_test.pt')
    tok_path = os.path.join(data_dir, f'{dataset_name}_tokenizer.pkl')

    if os.path.isfile(train_cache) and os.path.isfile(test_cache):
        logger.info(f'{dataset_name}: already processed, skipping.')
        return

    logger.info(f'Processing {dataset_name}...')
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Build tokenizer from all SMILES
    all_smiles = list(set(df_train['compound_iso_smiles']).union(
        set(df_test['compound_iso_smiles'])))

    if use_selfies:
        mol_strings = [MolTokenizer.smiles_to_selfies(s) for s in all_smiles]
        mol_strings = [s for s in mol_strings if s is not None]
    else:
        mol_strings = all_smiles

    vocabs = MolTokenizer.build_vocab(mol_strings, use_selfies=use_selfies)
    tokenizer = MolTokenizer(vocabs, use_selfies=use_selfies)

    with open(tok_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    # Also save as JSON (safe format)
    tok_json_path = tok_path.replace('.pkl', '.json')
    tokenizer.save_json(tok_json_path)
    logger.info(f'  Tokenizer: {len(tokenizer)} tokens (saved .pkl + .json)')

    # Build molecular graph cache
    graph_cache = build_graph_cache(all_smiles)

    # Compute descriptor normalization stats (for auxiliary task)
    desc_stats = None
    if include_aux:
        logger.info('  Computing descriptor statistics...')
        desc_stats = compute_descriptor_stats(all_smiles)
        stats_path = os.path.join(data_dir, f'{dataset_name}_desc_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(desc_stats, f)

    # Protein encoding
    if use_esm2:
        from dataset import extract_esm2_embeddings
        logger.info('  Extracting ESM-2 protein embeddings...')
        all_seqs = list(set(df_train['target_sequence']).union(
            set(df_test['target_sequence'])))
        seq_pairs = [(f'prot_{i}', seq) for i, seq in enumerate(all_seqs)]
        esm2_cache = extract_esm2_embeddings(seq_pairs)
        seq_to_emb = {seq: esm2_cache[f'prot_{i}'] for i, seq in enumerate(all_seqs)}

    # Process each split
    for split, df, cache_path in [('train', df_train, train_cache),
                                   ('test', df_test, test_cache)]:
        drugs = list(df['compound_iso_smiles'])
        target_smiles = list(df['target_smiles'])
        affinities = np.array(df['affinity'])

        # Protein features
        if use_esm2:
            proteins = np.array([seq_to_emb[s] for s in df['target_sequence']])
        else:
            proteins = np.array([encode_protein(s) for s in df['target_sequence']])

        # Tokenize target SMILES
        if use_selfies:
            mol_strs = [MolTokenizer.smiles_to_selfies(s) or '' for s in target_smiles]
        else:
            mol_strs = target_smiles

        drug_tokens = [torch.LongTensor(tokenizer.encode(s)) for s in mol_strs]

        build_dataset(
            drugs=drugs,
            drug_tokens=drug_tokens,
            proteins=proteins,
            affinities=affinities,
            graph_cache=graph_cache,
            cache_path=cache_path,
            morgan_fp_dim=morgan_fp_dim,
            include_descriptors=include_aux,
            include_funcgroups=include_aux,
            desc_stats=desc_stats,
        )
        logger.info(f'  {split} done.')


if __name__ == '__main__':
    cfg = ModelConfig()
    datasets = ['davis', 'kiba', 'bindingdb']

    use_esm2 = '--esm2' in sys.argv
    no_aux = '--no-aux' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if args:
        datasets = [args[0]]

    for ds in datasets:
        prepare_dataset(
            ds,
            use_selfies=cfg.use_selfies,
            use_esm2=use_esm2,
            include_aux=not no_aux,
            morgan_fp_dim=cfg.morgan_fp_dim if cfg.use_morgan_fp else 0,
        )

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    logger.info('Data preparation complete.')
