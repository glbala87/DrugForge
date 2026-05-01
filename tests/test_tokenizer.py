"""Tests for DrugForge tokenizer."""

import json
import os
import tempfile

import pytest

from tokenizer import MolTokenizer


@pytest.fixture
def tokenizer():
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O', 'CC(C)CC']
    vocabs = MolTokenizer.build_vocab(smiles)
    return MolTokenizer(vocabs)


class TestTokenizer:
    def test_vocab_size(self, tokenizer):
        assert len(tokenizer) > 4  # at least control tokens

    def test_control_tokens(self, tokenizer):
        assert tokenizer.CONTROL_TOKENS == ('[BOS]', '[EOS]', '[PAD]', '[UNK]')

    def test_round_trip(self, tokenizer):
        smiles = ['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O']
        for smi in smiles:
            encoded = tokenizer.encode(smi)
            decoded = tokenizer.decode([encoded])[0]
            assert decoded == smi, f"Round-trip failed: {smi} -> {decoded}"

    def test_bos_eos_framing(self, tokenizer):
        enc = tokenizer.encode("CCO")
        assert enc[0] == tokenizer.bos_id
        assert enc[-1] == tokenizer.eos_id

    def test_unknown_token(self, tokenizer):
        enc = tokenizer.encode("Xe")
        assert tokenizer.unk_id in enc

    def test_json_roundtrip(self, tokenizer):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            tokenizer.save_json(path)
            loaded = MolTokenizer.load_json(path)
            assert len(loaded) == len(tokenizer)
            # Verify encoding matches
            enc1 = tokenizer.encode("CCO")
            enc2 = loaded.encode("CCO")
            assert enc1 == enc2
        finally:
            os.unlink(path)

    def test_json_file_format(self, tokenizer):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            tokenizer.save_json(path)
            with open(path) as f:
                data = json.load(f)
            assert 'vocab' in data
            assert 'use_selfies' in data
            # Control tokens should NOT be in vocab list
            for ctrl in MolTokenizer.CONTROL_TOKENS:
                assert ctrl not in data['vocab']
        finally:
            os.unlink(path)
