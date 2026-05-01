"""Tests for DrugForge inference input validation."""

import pytest

from infer import validate_protein_sequence, validate_smiles


class TestProteinValidation:
    def test_valid_sequence(self):
        result = validate_protein_sequence("MKTAYIAKQRQISFVKSH")
        assert result == "MKTAYIAKQRQISFVKSH"

    def test_lowercase_normalized(self):
        result = validate_protein_sequence("mktayiakqrqisfvksh")
        assert result == "MKTAYIAKQRQISFVKSH"

    def test_valid_with_x(self):
        result = validate_protein_sequence("MKTAYIAKXQRQISFVKSH")
        assert "X" in result

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_protein_sequence("")

    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_protein_sequence("MKTAY12345IAKQRQ")

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            validate_protein_sequence("MKT")


class TestSmilesValidation:
    def test_valid_smiles(self):
        result = validate_smiles("CCO")
        assert result  # canonical form

    def test_canonical_output(self):
        result = validate_smiles("C(O)C")
        assert result == "CCO"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_smiles("")

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_smiles("XYZ_NOT_SMILES")
