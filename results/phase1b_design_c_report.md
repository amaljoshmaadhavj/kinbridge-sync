# Phase 1b Design C — Implementation Report

## Summary

Implemented the complete Phase 1b calibration pipeline using **Design C: dedicated independent negatives**. This resolves the fundamental methodological issue that the original pilot (19 a1↔a2 reissue pairs) contains zero negative pairs, making FMR undefined, alpha unestimable, and tau-star uncomputable.

## Design C Architecture

```
FSR (19 reissue pairs)          FMR (200 dedicated negative trials)
     │                                    │
     ▼                                    ▼
compute_fsr_curve()              compute_fmr_curve()
     │                                    │
     ▼                                    ▼
fit_beta() ──────────── fit_params_design_c() ──────────── fit_alpha()
                                                              │
                                                              ▼
                                              calibrate_tau_star() → Algorithm 2
```

**Key principle**: FSR and FMR are estimated from **independent data sources** and never mixed.

## New Files Created

### `pilot/run_negative_trials.py`
- Generates 200 dedicated negative trials per temperature (400 total, 800 action generations)
- Balances across all C(13,2) = 78 intent pairs with fair distribution
- Each trial: two actions on **different** intents from the same model
- Validates: different intents, both actions succeeded, no model call reuse
- Deterministic with seed=42

### `kinbridge_math/fmr_from_negatives.py`
- `compute_fmr_curve()`: FMR = |{distinct intents merged}| / |{intent pairs}|
  - Denominator: 200 (all intent pairs from dedicated negatives)
  - Uses Wilson 95% CI
- `compute_fsr_curve()`: FSR = |{same intent not merged}| / |{re-issued intents}|
  - Denominator: 19 (only a1↔a2 reissue pairs)
  - Uses Wilson 95% CI
- `load_fsr_from_reissues()`: Extracts only the 19 reissue pairs from pilot
- `load_negative_trials()`: Loads dedicated negative trial JSON files

### `kinbridge_math/calibrate.py` (rewritten)
- Accepts separate data sources: `--input` (pilot for FSR), `--fmr-input` (negatives for FMR)
- Temperature matching between pilot and negative files
- Outputs to `results/phase1b/`:
  - `calibration_summary.json`
  - `fmr_sweep.csv` (FMR from dedicated negatives)
  - `fsr_sweep.csv` (FSR from original reissues)
  - `alpha_beta_fits.csv`
  - `tau_star_results.csv`

## Test Coverage

**102 tests passing** including:
- Negative trial validation (intent, reuse, failed action detection)
- Intent pair selection (coverage, even distribution, determinism)
- FMR computation (denominator=200, 101 threshold points)
- FSR computation (denominator=19, 101 threshold points)
- Design C calibration (separate data sources, tau-star computation)
- Original pilot integrity (19 pairs, temperatures)

## Execution Commands

```bash
# Step 1: Generate 800 negative actions (requires Ollama running)
python -m pilot.run_negative_trials \
    --temperature 0.0 --n-trials 200 --seed 42 \
    --output results/raw_fmr/neg_trials_temp0.0.json

python -m pilot.run_negative_trials \
    --temperature 0.7 --n-trials 200 --seed 42 \
    --output results/raw_fmr/neg_trials_temp0.7.json

# Step 2: Run calibration
python -m kinbridge_math.calibrate \
    --input pilot/raw/pilot_temp0.0_20260916T160848Z.json \
            pilot/raw/pilot_temp0.7_20260916T162421Z.json \
    --fmr-input results/raw_fmr/neg_trials_temp0.0.json \
                 results/raw_fmr/neg_trials_temp0.7.json
```

## Files Modified

| File | Change |
|------|--------|
| `pilot/run_negative_trials.py` | Fixed `validate_trial` IndexError for empty tool_calls list |
| `kinbridge_math/calibrate.py` | Full rewrite for Design C (separate FSR/FMR sources) |
| `tests/test_pilot.py` | Replaced old TestCalibrate with TestCalibrateDesignC + TestNegativeTrials |
