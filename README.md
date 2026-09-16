# Kinbridge-Sync

Experimental validation pipeline for the Kinbridge-Sync research paper.

## Repository structure

```
kinbridge-sync/
├── config/                 # All experiment parameters (no hard-coded values)
│   ├── pilot/              # Phase 1: pilot study config + prompts
│   ├── experiment/         # Phase 5: factorial + ablation configs
│   ├── outage/             # Gilbert-Elliott model parameters
│   ├── tools/              # Tool schemas (JSON-schema for Ollama)
│   └── testbed/            # Docker / network testbed config
├── core/                   # Shared utilities (config loader, seed control)
├── pilot/                  # Phase 1: pilot study (prompt sweep, divergence)
├── math/                   # Phase 1b: Algorithm 2, Δq measurement
├── experiments/            # Phase 3-5: testbed, experiment runner, analysis
├── results/                # Raw CSV/JSON outputs (git-tracked .gitkeep only)
├── requirements.txt
└── .gitignore
```

## Reproducibility guarantees

- **Deterministic seeds**: all random state flows through `core.seeds.seed_everything()`.
- **Configuration-driven**: every experiment parameter lives in `config/` YAML files, never hard-coded in scripts.
- **Raw outputs**: CSV and JSON results go to `results/` — analysis scripts never overwrite raw data.
- **Separate analysis**: experiment runners and analysis scripts are always separate files.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Experiment phases

| Phase | Directory | Produces |
|-------|-----------|----------|
| 0 | — | Working environment (WSL2, Docker, Ollama) |
| 1 | `pilot/` | Table II, raw (τ, FMR, FSR) samples |
| 1b | `math/` | Validated τ*, Δq, Figure 2 |
| 2 | `tool_world/` | FastAPI tool endpoints + effect ledger |
| 3 | `experiments/` | Docker testbed + outage injection |
| 4 | `reintegration_service/` | Algorithm 1 + baselines (feature-flagged) |
| 5 | `experiments/` | Tables III-VI, final figures |

## Configuration

All YAML configs are loaded via `core.config.load_config()`:

```python
from core.config import load_config
cfg = load_config("pilot/pilot_config.yaml")  # relative to config/
```

## License

Research use only.
