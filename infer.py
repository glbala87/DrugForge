"""DrugForge Interactive Inference — generate drugs for a specific protein target.

Given a protein amino acid sequence and a reference drug SMILES,
generates novel drug molecules conditioned on the protein.

Usage:
    # Command line
    python infer.py --protein "MKTAYIAKQ..." --smiles "CCO" --model saved_models/drugforge_davis.pth

    # With a FASTA file
    python infer.py --fasta target.fasta --smiles "CCO" --model saved_models/drugforge_davis.pth

    # Specify affinity target and number of molecules
    python infer.py --protein "MKTAY..." --smiles "CCO" --affinity 7.5 --n-molecules 50

    # Python API
    from infer import DrugForgeInference
    engine = DrugForgeInference("saved_models/drugforge_davis.pth", "data/davis_tokenizer.pkl")
    results = engine.generate_for_target(
        protein_sequence="MKTAYIAKQRQISFVKSH...",
        reference_smiles="CC(=O)Oc1ccccc1C(=O)O",
        affinity_target=7.0,
        n_molecules=20,
    )
"""

import argparse
import logging
import os
import pickle
import re
import sys
from typing import List, Optional

import numpy as np
import torch
from torch_geometric import data as DATA
from torch_geometric.data import Batch

from config import ModelConfig
from dataset import (compute_atom_descriptors, compute_bond_descriptors,
                     encode_protein, smiles_to_graph, compute_morgan_fp)
from model import DrugForge

logger = logging.getLogger(__name__)


_VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')
_PROTEIN_RE = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYX]+$')


def validate_protein_sequence(sequence: str) -> str:
    """Validate and clean a protein amino acid sequence."""
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("Protein sequence is empty")
    if not _PROTEIN_RE.match(seq):
        invalid = set(seq) - _VALID_AA - {'X'}
        raise ValueError(f"Invalid amino acid characters: {invalid}")
    if len(seq) < 10:
        raise ValueError(f"Protein sequence too short ({len(seq)} residues, min 10)")
    return seq


def validate_smiles(smiles: str) -> str:
    """Validate a SMILES string using RDKit."""
    from rdkit import Chem
    smi = smiles.strip()
    if not smi:
        raise ValueError("SMILES string is empty")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


class DrugForgeInference:
    """High-level inference API for DrugForge.

    Load a trained model and generate drug molecules for any protein target.
    """

    def __init__(self, model_path: str, tokenizer_path: str,
                 device: str = 'cpu', cfg: ModelConfig = None):
        self.device = torch.device(device)
        self.cfg = cfg or ModelConfig()

        # Load tokenizer (prefer JSON, fall back to pickle)
        from tokenizer import MolTokenizer
        if tokenizer_path.endswith('.json'):
            self.tokenizer = MolTokenizer.load_json(tokenizer_path)
        elif os.path.isfile(tokenizer_path.replace('.pkl', '.json')):
            self.tokenizer = MolTokenizer.load_json(tokenizer_path.replace('.pkl', '.json'))
        else:
            logger.warning(f'Loading tokenizer from pickle (consider converting to JSON): {tokenizer_path}')
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)

        # Load model
        self.model = DrugForge(self.tokenizer, self.cfg)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        state = ckpt['model'] if 'model' in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f'DrugForge loaded | {sum(p.numel() for p in self.model.parameters()):,} params | {device}')

    def _prepare_input(self, protein_sequence: str, reference_smiles: str,
                       affinity_target: float = 7.0) -> DATA.Data:
        """Convert raw protein sequence + SMILES to a PyG Data object."""
        # Validate inputs
        protein_sequence = validate_protein_sequence(protein_sequence)
        reference_smiles = validate_smiles(reference_smiles)

        # Build molecular graph from reference SMILES
        graph = smiles_to_graph(reference_smiles)
        if graph is None:
            raise ValueError(f"Could not build graph for SMILES: {reference_smiles}")

        # Encode protein sequence as physicochemical features
        protein_feats = encode_protein(protein_sequence)

        # Tokenize reference SMILES (decoder target — not used during generation)
        token_ids = self.tokenizer.encode(reference_smiles)
        max_len = min(len(token_ids), self.cfg.max_gen_len)
        padded = torch.full((self.cfg.max_gen_len,), self.tokenizer.pad_id, dtype=torch.long)
        padded[:max_len] = torch.LongTensor(token_ids[:max_len])

        # Build PyG Data object
        data = DATA.Data(
            x=torch.from_numpy(graph['x']),
            edge_index=torch.from_numpy(graph['edge_index']),
            edge_attr=torch.from_numpy(graph['edge_attr']),
            y=torch.tensor([affinity_target], dtype=torch.float),
        )
        data.target = torch.from_numpy(protein_feats).unsqueeze(0).float()
        data.target_seq = padded.unsqueeze(0)
        data.__setitem__('c_size', torch.tensor([graph['num_atoms']], dtype=torch.long))

        # Morgan fingerprint (if model uses it)
        if self.cfg.use_morgan_fp:
            fp = compute_morgan_fp(reference_smiles, n_bits=self.cfg.morgan_fp_dim)
            data.__setitem__('morgan_fp', torch.from_numpy(fp).unsqueeze(0))

        return data

    @torch.no_grad()
    def generate_for_target(
        self,
        protein_sequence: str,
        reference_smiles: str,
        affinity_target: float = 7.0,
        n_molecules: int = 10,
        random_sample: bool = True,
        temperature: float = 1.0,
    ) -> dict:
        """Generate novel drug molecules for a specific protein target.

        Args:
            protein_sequence: amino acid sequence of the target protein
            reference_smiles: a known drug SMILES as structural seed
            affinity_target: desired binding affinity (pKd scale, higher = stronger)
            n_molecules: number of molecules to generate
            random_sample: True=diverse sampling, False=greedy (less diverse)
            temperature: sampling temperature (higher=more diverse, lower=more conservative)

        Returns:
            dict with keys:
                'generated': list of generated SMILES
                'valid': list of valid (RDKit-parseable) SMILES
                'unique': list of unique valid SMILES
                'novel': list of unique SMILES different from reference
                'stats': dict with validity/uniqueness/novelty ratios
        """
        # Prepare input data
        data = self._prepare_input(protein_sequence, reference_smiles, affinity_target)

        # Replicate for batch generation
        batch = Batch.from_data_list([data] * n_molecules).to(self.device)

        # Generate
        token_ids = self.model.generate(batch, random_sample=random_sample)
        generated = self.tokenizer.decode(token_ids)

        # Validate with RDKit
        from rdkit import Chem
        valid = []
        for smi in generated:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                valid.append(canonical)

        unique = list(set(valid))
        ref_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(reference_smiles), isomericSmiles=True)
        novel = [s for s in unique if s != ref_canonical]

        stats = {
            'total_generated': len(generated),
            'valid_count': len(valid),
            'unique_count': len(unique),
            'novel_count': len(novel),
            'validity': len(valid) / len(generated) if generated else 0,
            'uniqueness': len(unique) / len(valid) if valid else 0,
            'novelty': len(novel) / len(unique) if unique else 0,
        }

        return {
            'generated': generated,
            'valid': valid,
            'unique': unique,
            'novel': novel,
            'stats': stats,
        }

    @torch.no_grad()
    def predict_interaction(self, protein_sequence: str, drug_smiles: str) -> dict:
        """Predict drug-target interaction, affinity, and mechanism.

        Works in any task_mode — returns all available predictions.

        Args:
            protein_sequence: amino acid sequence
            drug_smiles: drug SMILES string

        Returns:
            dict with 'affinity', 'dti_probability', 'moa_probability' (where available)
        """
        data = self._prepare_input(protein_sequence, drug_smiles, affinity_target=0.0)
        batch = Batch.from_data_list([data]).to(self.device)

        # Get prediction from the model's predict() method
        pred = self.model.predict(batch)

        result = {}
        mode = self.cfg.task_mode

        if mode == 'dta':
            result['affinity'] = pred.item()
        elif mode == 'dti':
            result['dti_probability'] = pred.item()
            result['interacts'] = pred.item() > 0.5
        elif mode == 'moa':
            result['moa_probability'] = pred.item()
            result['mechanism'] = 'activation' if pred.item() > 0.5 else 'inhibition'
        elif mode == 'multi':
            result['affinity'] = pred.item()

        return result

    @torch.no_grad()
    def screen_compounds(self, protein_sequence: str, smiles_list: list) -> list:
        """Screen multiple compounds against a protein target.

        Args:
            protein_sequence: target protein sequence
            smiles_list: list of SMILES to screen

        Returns:
            list of dicts with 'smiles' and prediction scores, sorted by score
        """
        results = []
        for smi in smiles_list:
            try:
                pred = self.predict_interaction(protein_sequence, smi)
                pred['smiles'] = smi
                results.append(pred)
            except Exception:
                continue

        # Sort by affinity (descending) or probability (descending)
        if self.cfg.task_mode == 'dta':
            results.sort(key=lambda x: x.get('affinity', 0), reverse=True)
        else:
            key = 'dti_probability' if self.cfg.task_mode == 'dti' else 'moa_probability'
            results.sort(key=lambda x: x.get(key, 0), reverse=True)

        return results


def read_fasta(path: str) -> str:
    """Read first sequence from a FASTA file."""
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq_parts:
                    break
                continue
            seq_parts.append(line)
    return ''.join(seq_parts)


def main():
    parser = argparse.ArgumentParser(
        description='DrugForge: Generate drug molecules for a specific protein target',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 20 drugs for a protein target
  python infer.py --protein "MKTAYIAKQRQISFVK..." --smiles "CCO" --n-molecules 20

  # Use a FASTA file for the protein
  python infer.py --fasta my_target.fasta --smiles "c1ccccc1" --affinity 8.0

  # Greedy generation (less diverse, more "confident")
  python infer.py --protein "MKTAY..." --smiles "CCO" --no-random-sample
        """,
    )

    # Input
    protein_group = parser.add_mutually_exclusive_group(required=True)
    protein_group.add_argument('--protein', type=str, help='Protein amino acid sequence')
    protein_group.add_argument('--fasta', type=str, help='Path to FASTA file')

    parser.add_argument('--smiles', type=str, required=True,
                        help='Reference drug SMILES (structural seed)')
    parser.add_argument('--affinity', type=float, default=7.0,
                        help='Target binding affinity on pKd scale (default: 7.0)')

    # Mode
    parser.add_argument('--mode', type=str, default='generate',
                        choices=['generate', 'predict', 'screen'],
                        help='generate: new drugs, predict: affinity/DTI/MoA, screen: rank list of drugs')
    parser.add_argument('--screen-file', type=str, default=None,
                        help='File with SMILES (one per line) to screen in --mode=screen')

    # Generation
    parser.add_argument('--n-molecules', type=int, default=20,
                        help='Number of molecules to generate (default: 20)')
    parser.add_argument('--no-random-sample', action='store_true',
                        help='Use greedy decoding instead of random sampling')

    # Model
    parser.add_argument('--model', type=str, default='saved_models/drugforge_davis.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='data/davis_tokenizer.pkl',
                        help='Path to tokenizer pickle')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Save generated SMILES to file')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    )

    # Get protein sequence
    if args.fasta:
        protein_seq = read_fasta(args.fasta)
        logger.info(f'Protein from FASTA: {len(protein_seq)} residues')
    else:
        protein_seq = args.protein
        logger.info(f'Protein sequence: {len(protein_seq)} residues')

    # Validate inputs early
    try:
        protein_seq = validate_protein_sequence(protein_seq)
        validate_smiles(args.smiles)
    except ValueError as e:
        logger.error(f'Input validation failed: {e}')
        return

    logger.info(f'Reference SMILES: {args.smiles}')
    logger.info(f'Target affinity: {args.affinity} (pKd)')
    logger.info(f'Generating: {args.n_molecules} molecules')

    # Load model
    engine = DrugForgeInference(args.model, args.tokenizer, device=args.device)

    # Dispatch based on mode
    if args.mode == 'predict':
        logger.info(f"Predicting for {args.smiles}")
        pred = engine.predict_interaction(protein_seq, args.smiles)
        logger.info("=" * 60)
        logger.info("  PREDICTION")
        logger.info("=" * 60)
        for k, v in pred.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")
        return

    if args.mode == 'screen':
        if not args.screen_file:
            logger.error("--screen-file required for screen mode")
            return
        with open(args.screen_file) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        logger.info(f"Screening {len(smiles_list)} compounds...")
        results = engine.screen_compounds(protein_seq, smiles_list)
        logger.info("=" * 60)
        logger.info(f"  TOP {min(20, len(results))} COMPOUNDS")
        logger.info("=" * 60)
        for i, r in enumerate(results[:20], 1):
            score_key = 'affinity' if engine.cfg.task_mode == 'dta' else (
                'dti_probability' if engine.cfg.task_mode == 'dti' else 'moa_probability')
            logger.info(f"  {i:3d}. [{r.get(score_key, 0):.3f}] {r['smiles'][:70]}")
        if args.output:
            with open(args.output, 'w') as f:
                for r in results:
                    f.write(f"{r['smiles']}\t{r.get('affinity', r.get('dti_probability', r.get('moa_probability', 0)))}\n")
            logger.info(f"Full ranking saved to {args.output}")
        return

    # Default: generate
    results = engine.generate_for_target(
        protein_sequence=protein_seq,
        reference_smiles=args.smiles,
        affinity_target=args.affinity,
        n_molecules=args.n_molecules,
        random_sample=not args.no_random_sample,
    )

    # Display results
    stats = results['stats']
    logger.info("=" * 60)
    logger.info("  RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Generated:  {stats['total_generated']}")
    logger.info(f"  Valid:       {stats['valid_count']} ({stats['validity']:.1%})")
    logger.info(f"  Unique:      {stats['unique_count']} ({stats['uniqueness']:.1%})")
    logger.info(f"  Novel:       {stats['novel_count']} ({stats['novelty']:.1%})")
    logger.info("=" * 60)

    if results['novel']:
        logger.info("Novel drug candidates:")
        for i, smi in enumerate(results['novel'][:20], 1):
            logger.info(f"    {i:3d}. {smi}")
    elif results['valid']:
        logger.info("Valid molecules (may match reference):")
        for i, smi in enumerate(results['unique'][:20], 1):
            logger.info(f"    {i:3d}. {smi}")
    else:
        logger.warning("No valid molecules generated.")
        logger.info("Tip: Train the model longer (500 epochs) for better results.")

    # Save to file
    if args.output and results['novel']:
        with open(args.output, 'w') as f:
            f.write('\n'.join(results['novel']) + '\n')
        logger.info(f"Saved {len(results['novel'])} novel SMILES to {args.output}")


if __name__ == '__main__':
    main()
