"""DrugForge evaluation metrics — independent implementations.

DTA metrics use scikit-learn and scipy where possible.
Generation metrics use RDKit for chemical validation.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    average_precision_score,
    r2_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    accuracy_score,
)
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem


# ═══════════════════════════════════════════════════════════════════════════
# DTA Prediction Metrics
# ═══════════════════════════════════════════════════════════════════════════

def mse(y_true, y_pred):
    """Mean Squared Error (via sklearn)."""
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true, y_pred):
    """Root Mean Squared Error (via sklearn)."""
    from sklearn.metrics import root_mean_squared_error
    return float(root_mean_squared_error(y_true, y_pred))


def pearson_r(y_true, y_pred):
    """Pearson correlation coefficient (via scipy)."""
    r, _ = pearsonr(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel())
    return float(r)


def spearman_rho(y_true, y_pred):
    """Spearman rank correlation (via scipy)."""
    rho, _ = spearmanr(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel())
    return float(rho)


def concordance_index(y_true, y_pred):
    """Concordance index — pairwise comparison approach.

    For each ordered pair (i,j) where y_true_i > y_true_j,
    checks if y_pred_i > y_pred_j (concordant) or equal (tied, counts 0.5).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)

    concordant = 0.0
    permissible = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            permissible += 1.0
            if y_true[i] > y_true[j]:
                if y_pred[i] > y_pred[j]:
                    concordant += 1.0
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
            else:
                if y_pred[j] > y_pred[i]:
                    concordant += 1.0
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5

    return float(concordant / permissible) if permissible > 0 else 0.0


def concordance_index_fast(y_true, y_pred):
    """Vectorized concordance index for large datasets.

    Uses broadcasting for O(n^2) memory but fast numpy execution.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # All pairs where true[i] > true[j] for i > j
    true_diff = y_true[:, None] - y_true[None, :]
    pred_diff = y_pred[:, None] - y_pred[None, :]

    # Only consider pairs where true values differ (lower triangle for i>j)
    valid = np.tril(np.abs(true_diff) > 0, k=-1)
    if valid.sum() == 0:
        return 0.0

    # For valid pairs: concordant if signs match, half credit for ties
    concordant = np.where(true_diff > 0, pred_diff > 0, pred_diff < 0).astype(float)
    tied = (pred_diff == 0).astype(float) * 0.5
    score = (concordant + tied) * valid

    return float(score.sum() / valid.sum())


def modified_r2(y_true, y_pred):
    """Modified r-squared (rm2) metric.

    Computes r2 * (1 - sqrt(|r2^2 - r0_2^2|)) where r0_2 is
    the squared correlation through the origin.
    Uses sklearn.r2_score as foundation.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    # Standard R² via sklearn
    r2 = r2_score(y_true, y_pred)

    # R² through origin: y = k * y_pred
    pred_sq_sum = np.sum(y_pred ** 2)
    if pred_sq_sum == 0:
        return 0.0
    k = np.sum(y_true * y_pred) / pred_sq_sum

    ss_res_0 = np.sum((y_true - k * y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r0_2 = 1.0 - ss_res_0 / ss_tot if ss_tot > 0 else 0.0

    # Pearson r-squared (correlation coefficient squared)
    r, _ = pearsonr(y_true, y_pred)
    r_sq = r ** 2

    return float(r_sq * (1.0 - np.sqrt(np.abs(r_sq ** 2 - r0_2 ** 2))))


def aupr_at_thresholds(y_true, y_pred, thresholds):
    """Average Precision at multiple binary thresholds (via sklearn)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    results = []
    for t in thresholds:
        binary = (y_true > t).astype(int)
        pos = binary.sum()
        if pos == 0 or pos == len(binary):
            results.append(0.0)
        else:
            results.append(float(average_precision_score(binary, y_pred)))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# SMILES Generation Metrics
# ═══════════════════════════════════════════════════════════════════════════

def validate_smiles(smiles: str) -> bool:
    """Check if SMILES represents a valid molecule via RDKit."""
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def canonicalize_smiles(smiles: str):
    """Return canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def generation_metrics(generated, reference=None):
    """Compute validity, uniqueness, novelty of generated molecules."""
    if not generated:
        return {'validity': 0.0, 'uniqueness': 0.0, 'novelty': 0.0}

    valid = [s for s in generated if validate_smiles(s)]
    validity = len(valid) / len(generated)
    unique = set(valid)
    uniqueness = len(unique) / len(valid) if valid else 0.0

    novelty = 1.0
    if reference is not None:
        ref_set = set(reference)
        novel = [s for s in unique if s not in ref_set]
        novelty = len(novel) / len(unique) if unique else 0.0

    return {'validity': validity, 'uniqueness': uniqueness, 'novelty': novelty}


# ═══════════════════════════════════════════════════════════════════════════
# Classification Metrics (DTI / MoA)
# ═══════════════════════════════════════════════════════════════════════════

def auroc(y_true, y_score):
    """Area Under ROC Curve (via sklearn)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def aupr_score(y_true, y_score):
    """Area Under Precision-Recall Curve (via sklearn)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def classification_metrics(y_true, y_score, threshold=0.5):
    """Compute full classification metrics for DTI/MoA tasks.

    Args:
        y_true: binary ground truth labels
        y_score: predicted probabilities
        threshold: decision threshold for accuracy/F1

    Returns:
        dict with AUROC, AUPR, accuracy, F1
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    y_pred = (y_score >= threshold).astype(int)

    return {
        'AUROC': auroc(y_true, y_score),
        'AUPR': aupr_score(y_true, y_score),
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
