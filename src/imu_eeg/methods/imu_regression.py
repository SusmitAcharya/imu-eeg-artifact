"""IMU-referenced adaptive regression: batch ridge (OLS-equivalent) and streaming RLS.
Both predict EEG artifact from lagged IMU channels and subtract the prediction.
RLS must converge to the batch ridge solution on stationary data -- tested in
tests/test_rls_matches_ols.py before this file is trusted for anything else.
"""
import numpy as np


def build_lagged(imu, max_lag):
    """(n, k) IMU -> (n, k*(max_lag+1)) with causal lags 0..max_lag (lag 0 = same sample)."""
    n, k = imu.shape
    cols = []
    for lag in range(max_lag + 1):
        shifted = np.zeros((n, k))
        if lag == 0:
            shifted[:] = imu
        else:
            shifted[lag:] = imu[:-lag]
        cols.append(shifted)
    return np.hstack(cols)


def fit_ridge(X, Y, lam):
    """Standardised ridge regression, closed form. Returns (W, mu_x, sd_x, mu_y)."""
    mu_x, sd_x = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu_x) / sd_x
    mu_y = Y.mean(0)
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    W = np.linalg.solve(A, Xs.T @ (Y - mu_y))
    return W, mu_x, sd_x, mu_y


def predict_ridge(X, W, mu_x, sd_x, mu_y):
    return ((X - mu_x) / sd_x) @ W + mu_y


def clean_batch(eeg, imu, max_lag=32, lam=100.0, train_frac=0.5):
    """Fit on the first train_frac of the recording, predict and subtract on the rest.
    Returns (cleaned_eeg_on_test, eeg_test, n_train). Fit-then-subtract on the SAME
    samples used for fitting is not reported anywhere -- that would be circular
    (see project notes on why in-sample residual correlation is meaningless)."""
    X = build_lagged(imu, max_lag)
    n_train = int(len(X) * train_frac)
    W, mu_x, sd_x, mu_y = fit_ridge(X[:n_train], eeg[:n_train], lam)
    pred_test = predict_ridge(X[n_train:], W, mu_x, sd_x, mu_y)
    cleaned = eeg[n_train:] - (pred_test - mu_y)
    return cleaned, eeg[n_train:], n_train


class RLSCleaner:
    """Streaming recursive least squares, one IMU-lag feature vector at a time.
    Causal only: every prediction uses IMU samples up to and including the
    current index, never future ones. This is the online-deployable form of
    the batch ridge above; test_rls_matches_ols.py checks they agree."""

    def __init__(self, n_features, n_outputs, lam=100.0, forgetting=0.999):
        self.n_features = n_features
        self.forgetting = forgetting
        self.P = np.eye(n_features) / lam
        self.W = np.zeros((n_features, n_outputs))
        self.mu_x = np.zeros(n_features)
        self.mu_y = np.zeros(n_outputs)
        self._n = 0

    def _standardise(self, x):
        self._n += 1
        self.mu_x += (x - self.mu_x) / self._n
        return x - self.mu_x

    def update_and_predict(self, x, y):
        """One step: predict from x (pre-update), then update weights toward y.
        Returns the artifact prediction made BEFORE seeing y (causal)."""
        xs = self._standardise(x)
        pred = xs @ self.W + self.mu_y
        Pf = self.P / self.forgetting
        k = Pf @ xs / (1.0 + xs @ Pf @ xs)
        err = (y - self.mu_y) - xs @ self.W
        self.W += np.outer(k, err)
        self.P = Pf - np.outer(k, xs @ Pf)
        self.mu_y += 0.01 * (y - self.mu_y)  # slow-adapting output mean
        return pred

    def run(self, eeg, imu, max_lag=32):
        X = build_lagged(imu, max_lag)
        preds = np.zeros_like(eeg)
        for i in range(len(X)):
            preds[i] = self.update_and_predict(X[i], eeg[i])
        return eeg - (preds - self.mu_y), preds