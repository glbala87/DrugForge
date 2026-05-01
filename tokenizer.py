"""DrugForge Molecular Tokenizer — state-machine parser for SMILES/SELFIES.

Uses a character-level state machine (no regex) to tokenize SMILES strings.
Supports optional SELFIES representation for guaranteed chemical validity.
"""

import json
from typing import List

import torch
from tqdm.auto import tqdm


# Two-letter organic subset atoms
_TWO_CHAR_ATOMS = {'Br', 'Cl', 'Si', 'Se', 'se', 'Te', 'te'}

# Recognized single-char organic atoms
_SINGLE_CHAR_ATOMS = set('BCNOPSFIcnosp')


class MolTokenizer:
    """Molecular string tokenizer using a state-machine parser."""

    CONTROL_TOKENS = ('[BOS]', '[EOS]', '[PAD]', '[UNK]')

    def __init__(self, vocab_set, use_selfies: bool = False):
        self.use_selfies = use_selfies
        ctrl = list(self.CONTROL_TOKENS)
        sorted_vocab = sorted(set(vocab_set) - set(ctrl), key=lambda t: (len(t), t))
        all_tokens = ctrl + sorted_vocab

        self.tokens = all_tokens
        self.tok_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.id_to_tok = {i: t for t, i in self.tok_to_id.items()}

        # Convenience IDs
        self.s2i = self.tok_to_id   # alias for compatibility
        self.i2s = self.id_to_tok   # alias for compatibility

    def __len__(self):
        return len(self.tokens)

    @property
    def bos_id(self) -> int:
        return self.tok_to_id['[BOS]']

    @property
    def eos_id(self) -> int:
        return self.tok_to_id['[EOS]']

    @property
    def pad_id(self) -> int:
        return self.tok_to_id['[PAD]']

    @property
    def unk_id(self) -> int:
        return self.tok_to_id['[UNK]']

    # ------------------------------------------------------------------
    # JSON serialization (safe alternative to pickle)
    # ------------------------------------------------------------------

    def save_json(self, path: str):
        """Save tokenizer to a JSON file (safe, portable)."""
        data = {
            'vocab': [t for t in self.tokens if t not in self.CONTROL_TOKENS],
            'use_selfies': self.use_selfies,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> 'MolTokenizer':
        """Load tokenizer from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(set(data['vocab']), use_selfies=data.get('use_selfies', False))

    # ------------------------------------------------------------------
    # State-machine SMILES tokenizer (no regex)
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_smiles(smiles: str) -> List[str]:
        """Parse SMILES into tokens using character-level state machine."""
        tokens = []
        i, n = 0, len(smiles)

        while i < n:
            ch = smiles[i]

            # Bracketed atom/charge: [...]
            if ch == '[':
                end = smiles.index(']', i)
                tokens.append(smiles[i:end + 1])
                i = end + 1

            # Two-digit ring closure: %XX
            elif ch == '%' and i + 2 < n and smiles[i + 1].isdigit() and smiles[i + 2].isdigit():
                tokens.append(smiles[i:i + 3])
                i += 3

            # Stereochemistry: @@ or @
            elif ch == '@':
                if i + 1 < n and smiles[i + 1] == '@':
                    tokens.append('@@')
                    i += 2
                else:
                    tokens.append('@')
                    i += 1

            # Two-character organic atoms: Br, Cl, Si, Se, Te, etc.
            elif i + 1 < n and smiles[i:i + 2] in _TWO_CHAR_ATOMS:
                tokens.append(smiles[i:i + 2])
                i += 2

            # Everything else is a single character token
            else:
                tokens.append(ch)
                i += 1

        return tokens

    @staticmethod
    def _tokenize_selfies(selfies_str: str) -> List[str]:
        """Parse SELFIES into tokens (bracket-delimited)."""
        tokens = []
        i, n = 0, len(selfies_str)
        while i < n:
            if selfies_str[i] == '[':
                end = selfies_str.index(']', i)
                tokens.append(selfies_str[i:end + 1])
                i = end + 1
            else:
                i += 1  # skip non-bracket characters
        return tokens

    def tokenize(self, mol_string: str) -> List[str]:
        """Tokenize a molecular string."""
        if self.use_selfies:
            return self._tokenize_selfies(mol_string)
        return self._tokenize_smiles(mol_string)

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_vocab(mol_strings: List[str], use_selfies: bool = False) -> set:
        """Build vocabulary from a list of molecular strings."""
        vocab = set()
        tokenize = (MolTokenizer._tokenize_selfies if use_selfies
                    else MolTokenizer._tokenize_smiles)
        for s in tqdm(set(mol_strings), desc='Building vocab'):
            vocab.update(tokenize(s))
        return vocab

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, mol_string: str) -> List[int]:
        """Encode molecular string to token IDs with [BOS]/[EOS]."""
        raw_tokens = self.tokenize(mol_string)
        ids = [self.bos_id]
        for t in raw_tokens:
            ids.append(self.tok_to_id.get(t, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def decode(self, token_ids) -> List[str]:
        """Decode batch of token ID sequences to molecular strings."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        ctrl = set(self.CONTROL_TOKENS)
        results = []
        for seq in token_ids:
            parts = []
            for tid in seq:
                tok = self.id_to_tok.get(tid, '[UNK]')
                if tok == '[EOS]':
                    break
                if tok not in ctrl:
                    parts.append(tok)
            results.append(''.join(parts))
        return results

    # ------------------------------------------------------------------
    # SELFIES conversion utilities
    # ------------------------------------------------------------------

    @staticmethod
    def smiles_to_selfies(smiles: str) -> str:
        import selfies as sf
        return sf.encoder(smiles)

    @staticmethod
    def selfies_to_smiles(selfies_str: str) -> str:
        import selfies as sf
        return sf.decoder(selfies_str)
