"""Tests for DrugForge metrics module."""

import numpy as np
import pytest

from metrics import (
    mse, rmse, pearson_r, spearman_rho,
    concordance_index, concordance_index_fast,
    modified_r2, aupr_at_thresholds,
    validate_smiles, generation_metrics,
    classification_metrics,
)


@pytest.fixture
def perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    return y_true, y_pred


class TestRegressionMetrics:
    def test_mse(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert abs(mse(y_true, y_pred) - 0.022) < 1e-6

    def test_rmse(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert abs(rmse(y_true, y_pred) - 0.14832) < 1e-4

    def test_pearson_high(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert pearson_r(y_true, y_pred) > 0.99

    def test_spearman_perfect_rank(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert abs(spearman_rho(y_true, y_pred) - 1.0) < 1e-10

    def test_ci_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert abs(concordance_index(y_true, y_pred) - 1.0) < 1e-10

    def test_ci_fast_matches(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        ci = concordance_index(y_true, y_pred)
        ci_fast = concordance_index_fast(y_true, y_pred)
        assert abs(ci - ci_fast) < 1e-6

    def test_ci_reversed(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(concordance_index(y_true, y_pred) - 0.0) < 1e-10

    def test_modified_r2_range(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        rm = modified_r2(y_true, y_pred)
        assert 0 < rm < 1

    def test_aupr_at_thresholds(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        auprs = aupr_at_thresholds(y_true, y_pred, [2.5, 3.5])
        assert len(auprs) == 2
        assert all(0 <= a <= 1 for a in auprs)


class TestSmilesValidation:
    def test_valid_smiles(self):
        assert validate_smiles("CCO") is True
        assert validate_smiles("c1ccccc1") is True

    def test_invalid_smiles(self):
        assert validate_smiles("XYZ") is False


class TestGenerationMetrics:
    def test_generation_metrics(self):
        gen = generation_metrics(
            ["CCO", "c1ccccc1", "INVALID", "CCO"],
            reference=["CCO"],
        )
        assert abs(gen['validity'] - 0.75) < 1e-6
        assert abs(gen['uniqueness'] - 2/3) < 1e-6
        assert abs(gen['novelty'] - 0.5) < 1e-6

    def test_empty_generation(self):
        gen = generation_metrics([])
        assert gen['validity'] == 0.0


class TestClassificationMetrics:
    def test_basic_classification(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.6, 0.9])
        m = classification_metrics(y_true, y_score)
        assert 'AUROC' in m
        assert 'AUPR' in m
        assert 'Accuracy' in m
        assert 'F1' in m
        assert m['AUROC'] > 0.5
