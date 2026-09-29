# IMU-EEG artifact suppression: exported scripts (01 through 03)

Everything here matches what was verified end-to-end in chat against your real
12-file Emotiv dataset, in a fresh venv, on 2026-09-29. One real bug was
caught and fixed during this export (see below) that would have broken
`tests\test_rls_matches_ols.py` on your machine.

## What's included
- `scripts/01_inspect_data.py` -- first-pass CSV inspection and plots
- `scripts/01b_headers_alpha_feasibility.py` -- column-name/alpha/feasibility check
- `scripts/01c_coupling_check.py` -- coherence + blocked-CV coupling check
- `scripts/02_preprocess.py` -- band-pass, trim, save to data/interim/*.npz
- `scripts/03_run_methods.py` -- runs all 5 methods x 3 channel counts x 12 files
- `src/imu_eeg/` -- config, io, preprocess, subsample, metrics, methods/
- `tests/` -- test_preprocess.py, test_rls_matches_ols.py
- `docs/hypotheses.md` -- pre-specified H1/H2/H3 (LS, EBM, Music groups still
  need your input from the article's activity table -- see chat)
- `requirements.txt` -- rewritten as a minimal, tested dependency list
  (your version was a full `pip freeze` of ~100 packages, mostly unrelated
  Jupyter internals; this one has just what the code actually imports)

## What's NOT included
- Raw data (`data/raw/emotiv/...`) -- re-download from Mendeley as before
- `results/`, `data/interim/*.npz` -- regenerate by running the scripts
- `external/asrpy`, `data/raw/wearbci_repo` -- re-clone if needed
- `scripts/04_metrics.py`, `05_classify.py`, `06_figures.py`, `src/imu_eeg/classify.py`,
  `src/imu_eeg/stats.py` -- confirmed empty on your end, not yet written

## Two fixes applied in this export

1. **`tests/test_rls_matches_ols.py` had the wrong content.** Your zip's copy
   of this file contained a pasted copy of `03_run_methods.py`'s source instead
   of the actual RLS convergence/causality test. Filename was right, content
   was wrong. Restored to the correct test (verified: passes, prediction
   correlation 0.999 against batch ridge, causality confirmed).

2. **`held_out_r2` renamed to `imu_predictability_residual`, both in
   `src/imu_eeg/metrics.py` and its caller in `scripts/03_run_methods.py`,
   and its CSV column renamed to `residual_imu_r2_DIAGNOSTIC_ONLY`.** The
   function was never scoring "how good is this cleaning method" -- it
   re-fits a fresh IMU-to-cleaned-EEG regression and reports that fit's own
   held-out R2, so a method that destroys real EEG scores just as well as one
   that removes only artifact. The math wasn't broken, the interpretation was.
   `alpha_contrast` (H3) and `spectral_fidelity_D_PSD` (H1/H2) are the metrics
   to actually read; the renamed column is kept only as a clearly-labelled
   diagnostic, excluded from any ranking.

## To run
```
python -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python tests\test_preprocess.py
.venv\Scripts\python tests\test_rls_matches_ols.py
.venv\Scripts\python scripts\01_inspect_data.py
.venv\Scripts\python scripts\01b_headers_alpha_feasibility.py
.venv\Scripts\python scripts\01c_coupling_check.py
.venv\Scripts\python scripts\02_preprocess.py
.venv\Scripts\python scripts\03_run_methods.py
```
Expect: 3/3 preprocess tests pass, RLS prediction correlation ~0.999, both RLS
tests pass, then the same numbers you already saw in chat for 01b/01c/03.

Next: send the LS/EBM/Music activity-group assignment from the dataset
article, then `04_metrics.py` gets written against real data.
