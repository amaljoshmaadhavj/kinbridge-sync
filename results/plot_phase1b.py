"""Generate Phase 1b FMR/FSR calibration figure from verified CSV outputs.

Reads:
  results/phase1b/fmr_sweep.csv
  results/phase1b/fsr_sweep.csv
  results/phase1b/calibration_summary.json

Outputs:
  results/phase1b/fmr_fsr_calibration.png  (300 DPI)
  results/phase1b/fmr_fsr_curve.pdf         (paper convention)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).parent / "phase1b"
FMR_CSV = RESULTS_DIR / "fmr_sweep.csv"
FSR_CSV = RESULTS_DIR / "fsr_sweep.csv"
SUMMARY_JSON = RESULTS_DIR / "calibration_summary.json"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fmr = pd.read_csv(FMR_CSV)
    fsr = pd.read_csv(FSR_CSV)
    with open(SUMMARY_JSON) as f:
        summary = json.load(f)
    return fmr, fsr, summary


def get_fits(summary: dict) -> dict:
    """Extract alpha/beta fits per temperature from the summary JSON."""
    fits = {}
    for t in summary["temperatures"]:
        temp = t["temperature"]
        fits[temp] = {
            "alpha": t["fit_alpha"]["slope"],
            "alpha_r2": t["fit_alpha"]["r_squared"],
            "beta": t["fit_beta"]["slope"],
            "beta_r2": t["fit_beta"]["r_squared"],
            "tau_star": t["tau_star"]["tau_star"],
            "tau_star_method": t["tau_star"]["method_used"],
            "risk": t["tau_star"]["risk_at_tau_star"],
        }
    return fits


def plot_figure(
    fmr: pd.DataFrame,
    fsr: pd.DataFrame,
    fits: dict,
) -> plt.Figure:
    fig, (ax_fmr, ax_fsr) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    colors = {0.0: "#1f77b4", 0.7: "#ff7f0e"}
    linestyles = {0.0: "-", 0.7: "--"}
    temp_labels = {0.0: "T=0.0", 0.7: "T=0.7"}

    for temp in [0.0, 0.7]:
        fmr_t = fmr[fmr["temperature"] == temp].sort_values("tau")
        fsr_t = fsr[fsr["temperature"] == temp].sort_values("tau")

        c = colors[temp]
        ls = linestyles[temp]

        # --- FMR panel ---
        ax_fmr.plot(
            fmr_t["tau"], fmr_t["fmr"],
            color=c, linestyle=ls, linewidth=1.8,
            label=f"FMR {temp_labels[temp]}",
        )
        ax_fmr.fill_between(
            fmr_t["tau"], fmr_t["ci_low"], fmr_t["ci_high"],
            color=c, alpha=0.15,
        )

        # Fitted FMR curve for T=0.7: (1-tau)^alpha * exp(intercept)
        if temp == 0.7:
            alpha = fits[0.7]["alpha"]
            intercept_a = fits[0.7].get("alpha_intercept", 0.0)
            tau_grid = np.linspace(0.001, 0.999, 500)
            fmr_fitted = np.exp(intercept_a) * (1.0 - tau_grid) ** alpha
            ax_fmr.plot(
                tau_grid, fmr_fitted,
                color=c, linestyle=":", linewidth=1.2, alpha=0.7,
                label=f"Fitted (1−τ)^α  α={alpha:.3f}",
            )

        # --- FSR panel ---
        ax_fsr.plot(
            fsr_t["tau"], fsr_t["fsr"],
            color=c, linestyle=ls, linewidth=1.8,
            label=f"FSR {temp_labels[temp]}",
        )
        ax_fsr.fill_between(
            fsr_t["tau"], fsr_t["ci_low"], fsr_t["ci_high"],
            color=c, alpha=0.15,
        )

        # Fitted FSR curve for T=0.7: tau^beta * exp(intercept)
        if temp == 0.7:
            beta = fits[0.7]["beta"]
            intercept_b = fits[0.7].get("beta_intercept", 0.0)
            tau_grid = np.linspace(0.001, 0.999, 500)
            fsr_fitted = np.exp(intercept_b) * tau_grid ** beta
            ax_fsr.plot(
                tau_grid, fsr_fitted,
                color=c, linestyle=":", linewidth=1.2, alpha=0.7,
                label=f"Fitted τ^β  β={beta:.3f}",
            )

    # --- Tau-star annotation for T=0.7 ---
    tau_star_07 = fits[0.7]["tau_star"]
    beta_r2_07 = fits[0.7]["beta_r2"]

    # Mark tau* on FMR panel
    ax_fmr.axvline(
        tau_star_07, color="gray", linestyle="-.", linewidth=1.0, alpha=0.6,
    )
    ax_fmr.annotate(
        f"τ*={tau_star_07:.3f}\n(β R²={beta_r2_07:.2f})",
        xy=(tau_star_07, 0.85),
        xytext=(tau_star_07 + 0.08, 0.88),
        fontsize=8, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Mark tau* on FSR panel
    ax_fsr.axvline(
        tau_star_07, color="gray", linestyle="-.", linewidth=1.0, alpha=0.6,
    )
    ax_fsr.annotate(
        f"τ*={tau_star_07:.3f}\n(β R²={beta_r2_07:.2f})",
        xy=(tau_star_07, 0.25),
        xytext=(tau_star_07 + 0.08, 0.30),
        fontsize=8, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # --- Axes formatting ---
    for ax in (ax_fmr, ax_fsr):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Similarity Threshold (τ)", fontsize=11)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    ax_fmr.set_ylabel("Rate", fontsize=11)
    ax_fmr.set_title("False Match Rate (FMR)", fontsize=12, fontweight="bold")
    ax_fsr.set_title("False Separation Rate (FSR)", fontsize=12, fontweight="bold")

    # --- Legends ---
    ax_fmr.legend(fontsize=8, loc="upper right")
    ax_fsr.legend(fontsize=8, loc="upper left")

    # --- Suptitle ---
    fig.suptitle(
        "Phase 1b Design C Calibration — Empirical FMR and FSR Curves",
        fontsize=13, fontweight="bold", y=1.02,
    )

    fig.tight_layout()
    return fig


def main() -> None:
    fmr, fsr, summary = load_data()

    # Extract intercepts from the summary (stored in nested fit dicts)
    fits = get_fits(summary)
    for temp in [0.0, 0.7]:
        t_data = next(t for t in summary["temperatures"] if t["temperature"] == temp)
        fits[temp]["alpha_intercept"] = t_data["fit_alpha"]["intercept"]
        fits[temp]["beta_intercept"] = t_data["fit_beta"]["intercept"]

    fig = plot_figure(fmr, fsr, fits)

    # Save outputs
    png_path = RESULTS_DIR / "fmr_fsr_calibration.png"
    pdf_path = RESULTS_DIR / "fmr_fsr_curve.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- Verification ---
    fmr_pts = len(fmr)
    fsr_pts = len(fsr)
    print(f"Source: {FMR_CSV.name}, {FSR_CSV.name}, {SUMMARY_JSON.name}")
    print(f"Output: {png_path.name} (PNG 300 DPI), {pdf_path.name} (PDF)")
    print(f"Points plotted: FMR={fmr_pts}, FSR={fsr_pts} (empirical + CI bands)")
    print(f"Confidence intervals: Yes (Wilson 95%, shaded)")
    print(f"Tau-star marked: Yes (T=0.7 only, tau*={fits[0.7]['tau_star']:.4f})")
    print(f"Tau-star T=0.0: Not plotted (unestimable)")
    print(f"Fitted curves: T=0.7 only, labeled as models")
    print("No experimental data modified.")


if __name__ == "__main__":
    main()
