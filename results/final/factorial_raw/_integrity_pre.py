import hashlib
import json
import time
from pathlib import Path

from core.config import PROJECT_ROOT


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


paths = sorted((PROJECT_ROOT / "results" / "phase1b").glob("*"))
for rel in [
    "config/experiment/factorial.yaml",
    "config/experiment/ablation.yaml",
    "config/experiment/phase5_scenario.yaml",
    "client/buffer.py",
    "reintegration/reintegration_service.py",
    "reintegration/naive_retry.py",
    "reintegration/verify_before_retry.py",
    "tool_world/main.py",
    "tool_world/ledger.py",
    "pilot/key_schemes.py",
    "pilot/wilson.py",
]:
    paths.append(PROJECT_ROOT / rel)

manifest = {
    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "files": {
        str(p.relative_to(PROJECT_ROOT)): digest(p)
        for p in paths
        if p.exists() and p.is_file()
    },
}
out = PROJECT_ROOT / "results" / "final" / "factorial_raw" / "integrity_pre_run.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
print("manifest files:", len(manifest["files"]))
print("phase1b files:", sum(1 for k in manifest["files"] if k.startswith("results/phase1b")))
