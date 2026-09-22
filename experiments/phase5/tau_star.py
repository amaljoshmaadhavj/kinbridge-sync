"""Tau-star calibration bridge (B2 resolution).

Reads the calibrated tau* from the existing Phase 1b artifact

    results/phase1b/tau_star_results.csv

instead of hardcoding the value.  The loader validates:
  - the calibration context (artefact exists, parses)
  - the requested calibration temperature
  - numeric finite tau*
  - provenance path

T=0.0 is recorded in the artifact as ``unestimable`` (beta unestimable,
FSR=0); in that context ``load_tau_star(temperature=0.0)`` returns
``None`` and callers must fail closed / escalate per Phase 4 semantics.

Never substitutes a fallback threshold.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.config import PROJECT_ROOT


@dataclass(frozen=True)
class TauStarResult:
    """One row of the Phase 1b calibration artifact."""

    temperature: float
    tau_star: Optional[float]
    alpha: Optional[float]
    beta: Optional[float]
    method_used: str
    risk_at_tau_star: Optional[float]
    w_d: float
    w_m: float
    provenance: str

    @property
    def estimable(self) -> bool:
        """True when a finite tau* exists for this temperature."""
        return self.tau_star is not None


class CalibrationArtifactError(ValueError):
    """Raised when the calibration artifact is missing or malformed."""


def default_artifact_path() -> Path:
    """Return the canonical Phase 1b tau* artifact path."""
    return PROJECT_ROOT / "results" / "phase1b" / "tau_star_results.csv"


def _parse_optional(value: str) -> Optional[float]:
    """Parse an optional numeric cell; empty/NaN -> None."""
    v = value.strip()
    if not v:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def load_tau_star_csv(path: str | Path) -> dict[float, TauStarResult]:
    """Parse the Phase 1b tau* artifact into {temperature: TauStarResult}.

    Raises:
        CalibrationArtifactError: if the artifact is missing or malformed.
    """
    p = Path(path)
    if not p.exists():
        raise CalibrationArtifactError(
            f"Calibration artifact not found: {p} "
            "(expected results/phase1b/tau_star_results.csv)"
        )
    results: dict[float, TauStarResult] = {}
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "temperature" not in reader.fieldnames:
            raise CalibrationArtifactError(
                f"Malformed calibration artifact {p}: missing 'temperature' column"
            )
        for row in reader:
            try:
                temp = float(row["temperature"])
            except (KeyError, ValueError):
                raise CalibrationArtifactError(
                    f"Malformed calibration row in {p}: {row!r}"
                )
            results[temp] = TauStarResult(
                temperature=temp,
                tau_star=_parse_optional(row.get("tau_star", "")),
                alpha=_parse_optional(row.get("alpha", "")),
                beta=_parse_optional(row.get("beta", "")),
                method_used=(row.get("method_used", "") or "").strip(),
                risk_at_tau_star=_parse_optional(row.get("risk_at_tau_star", "")),
                w_d=float(row.get("w_d", 4.0)),
                w_m=float(row.get("w_m", 1.0)),
                provenance=str(p.resolve()),
            )
    if not results:
        raise CalibrationArtifactError(
            f"Calibration artifact {p} contains no temperature rows"
        )
    return results


def load_tau_star(
    temperature: float,
    path: str | Path | None = None,
) -> tuple[Optional[float], TauStarResult]:
    """Return (tau_star, provenance result) for a calibration temperature.

    Returns ``(None, row)`` when the temperature is not calibratable
    (T=0.0 ``unestimable``), and raises for an unknown temperature or a
    missing/malformed artifact.  The returned float is always finite.

    Raises:
        CalibrationArtifactError: artifact missing/malformed, or the
            temperature has no calibration row.
    """
    artifact = path if path is not None else default_artifact_path()
    results = load_tau_star_csv(artifact)
    near = None
    for temp in results:
        if abs(temp - temperature) < 1e-9:
            near = temp
            break
    if near is None:
        raise CalibrationArtifactError(
            f"No calibration row for temperature {temperature!r} in {artifact}; "
            f"available temperatures: {sorted(results)}"
        )
    row = results[near]
    if not row.estimable:
        return None, row
    assert row.tau_star is not None
    return row.tau_star, row


def calibration_provenance(
    temperature: float,
    path: str | Path | None = None,
) -> dict[str, object]:
    """Metadata describing where tau* came from (for episode records)."""
    tau, row = load_tau_star(temperature, path)
    return {
        "tau_star": tau,
        "tau_star_source": row.provenance,
        "calibration_temperature": row.temperature,
        "calibration_status": row.method_used if row.estimable else "unestimable",
    }


def phi_tau_star_provider(
    temperature: float,
    scenario_temperature: float,
    path: str | Path | None = None,
):
    """Return a Phase-4-style tau* provider for a calibration context.

    The provider returns the artifact value for the calibration
    temperature, or ``None`` (fail closed) when unestimable.  The
    scenario temperature must match the calibration context; a mismatch
    is treated the same as an unavailable tau* (never substituted).

    The returned callable has the same signature as the Phase 4
    ``TauStarProvider`` hook.
    """
    if abs(temperature - scenario_temperature) > 1e-9:
        def _mismatch() -> Optional[float]:
            return None
        return _mismatch
    tau, row = load_tau_star(temperature, path)

    def _provider() -> Optional[float]:
        return tau

    return _provider