"""RLS must converge to the batch ridge (OLS-with-ridge) solution on stationary,
linearly-related synthetic data. Run:  .venv\\Scripts\\python tests\\test_rls_matches_ols.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from imu_eeg.methods.imu_regression import RLSCleaner, build_lagged, fit_ridge, predict_ridge


def make_stationary(n=6000, k=6, max_lag=8, seed=0):
    rng = np.random.default_rng(seed)
    imu = rng.normal(size=(n, k))
    X = build_lagged(imu, max_lag)
    true_W = rng.normal(scale=0.3, size=(X.shape[1], 4))
    eeg = X @ true_W + rng.normal(scale=0.05, size=(n, 4))
    return imu, eeg, max_lag


def test_rls_converges_to_batch_ridge():
    imu, eeg, max_lag = make_stationary()
    X = build_lagged(imu, max_lag)
    lam = 50.0
    W_batch, mu_x, sd_x, mu_y = fit_ridge(X, eeg, lam)

    rls = RLSCleaner(n_features=X.shape[1], n_outputs=eeg.shape[1], lam=lam, forgetting=1.0)
    for i in range(len(X)):
        rls.update_and_predict(X[i], eeg[i])

    # RLS has no explicit standardisation (unlike the batch fit), so compare
    # PREDICTIONS on held-out-style data, not raw weights.
    rng = np.random.default_rng(99)
    test_imu = rng.normal(size=(500, imu.shape[1]))
    Xt = build_lagged(test_imu, max_lag)
    pred_batch = predict_ridge(Xt, W_batch, mu_x, sd_x, mu_y)
    xs_t = Xt - rls.mu_x
    pred_rls = xs_t @ rls.W + rls.mu_y
    corr = np.corrcoef(pred_batch.ravel(), pred_rls.ravel())[0, 1]
    rel_err = np.linalg.norm(pred_batch - pred_rls) / np.linalg.norm(pred_batch)
    print(f"  prediction correlation: {corr:.4f}   relative error: {rel_err:.4f}")
    assert corr > 0.9, f"RLS predictions should track batch ridge closely, corr={corr:.4f}"


def test_rls_is_causal():
    """Perturbing IMU AFTER sample i must not change the prediction made AT sample i."""
    rng = np.random.default_rng(1)
    imu, eeg, max_lag = make_stationary(seed=2)
    n_check = 200
    X = build_lagged(imu, max_lag)

    rls_a = RLSCleaner(X.shape[1], eeg.shape[1], lam=50.0)
    preds_a = [rls_a.update_and_predict(X[i], eeg[i]).copy() for i in range(n_check)]

    imu2 = imu.copy()
    imu2[n_check + 5:] = rng.normal(size=imu2[n_check + 5:].shape)  # change the future only
    X2 = build_lagged(imu2, max_lag)
    rls_b = RLSCleaner(X.shape[1], eeg.shape[1], lam=50.0)
    preds_b = [rls_b.update_and_predict(X2[i], eeg[i]).copy() for i in range(n_check)]

    for i in range(n_check):
        assert np.allclose(preds_a[i], preds_b[i], atol=1e-10), f"causality broken at sample {i}"


if __name__ == "__main__":
    test_rls_converges_to_batch_ridge()
    test_rls_is_causal()
    print("Both RLS tests passed")
