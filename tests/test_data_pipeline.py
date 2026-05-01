"""Tests for DrugForge data pipeline."""

import os

import numpy as np
import pytest

from dataset import (
    compute_atom_descriptors, compute_bond_descriptors,
    encode_protein, smiles_to_graph, build_graph_cache,
    compute_morgan_fp, compute_mol_descriptors,
    compute_functional_groups, PROTEIN_MAX_LEN,
)


class TestAtomDescriptors:
    def test_shape(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        atom = mol.GetAtomWithIdx(0)
        ad = compute_atom_descriptors(atom)
        assert ad.shape == (17,)

    def test_finite(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")
        for atom in mol.GetAtoms():
            ad = compute_atom_descriptors(atom)
            assert np.all(np.isfinite(ad))

    def test_normalized_range(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        for atom in mol.GetAtoms():
            ad = compute_atom_descriptors(atom)
            assert ad.max() <= 1.5


class TestBondDescriptors:
    def test_shape(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        bond = mol.GetBondWithIdx(0)
        bd = compute_bond_descriptors(bond)
        assert bd.shape == (6,)


class TestProteinEncoding:
    def test_shape(self):
        prot = encode_protein("ACDEFGHIKLMNPQRSTVWY")
        assert prot.shape == (PROTEIN_MAX_LEN, 8)

    def test_nonzero(self):
        prot = encode_protein("ACDEFGHIKLMNPQRSTVWY")
        assert prot[:20].sum() > 0

    def test_padding_zeros(self):
        prot = encode_protein("ACD")
        assert prot[3:].sum() == 0.0


class TestGraphConstruction:
    def test_valid_smiles(self):
        g = smiles_to_graph("CCO")
        assert g is not None
        assert g['x'].shape[1] == 17
        assert g['edge_attr'].shape[1] == 6
        assert g['edge_index'].shape[0] == 2

    def test_invalid_smiles(self):
        g = smiles_to_graph("INVALID_SMILES")
        assert g is None

    def test_single_atom(self):
        g = smiles_to_graph("[Na]")
        assert g is not None
        assert g['num_atoms'] == 1

    def test_graph_cache(self):
        smiles = ["CCO", "c1ccccc1", "INVALID"]
        cache = build_graph_cache(smiles)
        assert "CCO" in cache
        assert "c1ccccc1" in cache
        assert "INVALID" not in cache


class TestMorganFP:
    def test_shape(self):
        fp = compute_morgan_fp("CCO", n_bits=256)
        assert fp.shape == (256,)
        assert fp.dtype == np.float32

    def test_invalid_returns_zeros(self):
        fp = compute_morgan_fp("INVALID", n_bits=256)
        assert fp.sum() == 0.0


class TestMolDescriptors:
    def test_returns_array(self):
        desc = compute_mol_descriptors("CCO")
        assert desc.ndim == 1
        assert len(desc) > 0

    def test_invalid_returns_zeros(self):
        desc = compute_mol_descriptors("INVALID")
        assert desc.sum() == 0.0


class TestFunctionalGroups:
    def test_returns_binary(self):
        fg = compute_functional_groups("CC(=O)O")  # acetic acid
        assert set(np.unique(fg)).issubset({0.0, 1.0})

    def test_invalid_returns_zeros(self):
        fg = compute_functional_groups("INVALID")
        assert fg.sum() == 0.0
