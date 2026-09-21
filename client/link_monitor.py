"""Client-side network link health monitor (Jacobson/Karels EWMA + hysteresis FSM).

Estimates round-trip time and its deviation using the EWMA equations
from Section 3.2 of Complete-Research.md, and tracks link state
(UP / DOWN) with a hysteresis FSM to avoid single-packet flapping.

EWMA equations (Jacobson/Karels):
    r_hat_t  = alpha * r_t  + (1 - alpha) * r_hat_{t-1}
    d_hat_t  = beta  * |r_t - r_hat_t| + (1 - beta)  * d_hat_{t-1}
    RTO      = r_hat + 4 * d_hat

Defaults (from config/testbed/network.yaml):
    alpha               = 0.125
    beta                = 0.25
    timeout_threshold   = 5    (consecutive timeouts → DOWN)
    recovery_threshold  = 3    (consecutive successes → UP)

References:
    Markdown/Complete-Research.md  §3.2  (EWMA equations, RTO)
    Markdown/Guide.md              lines 150-173  (Phase 2 spec)
    Markdown/Work .md              lines 1484-1522 (equations)
    config/testbed/network.yaml    (defaults)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "alpha": 0.125,
    "beta": 0.25,
    "timeout_threshold": 5,
    "recovery_threshold": 3,
}

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "testbed" / "network.yaml"


def _load_defaults() -> dict[str, Any]:
    """Load defaults from network.yaml, falling back to hard-coded values."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        lm = cfg.get("link_monitor", {})
        return {
            "alpha": float(lm.get("alpha", _DEFAULTS["alpha"])),
            "beta": float(lm.get("beta", _DEFAULTS["beta"])),
            "timeout_threshold": int(lm.get("timeout_threshold", _DEFAULTS["timeout_threshold"])),
            "recovery_threshold": int(lm.get("recovery_threshold", _DEFAULTS["recovery_threshold"])),
        }
    except Exception:
        return dict(_DEFAULTS)


# ---------------------------------------------------------------------------
# LinkMonitor
# ---------------------------------------------------------------------------

class LinkMonitor:
    """Jacobson/Karels EWMA link-health monitor with UP/DOWN hysteresis FSM.

    Args:
        alpha:  EWMA smoothing weight for RTT estimate  (default from config).
        beta:   EWMA smoothing weight for deviation      (default from config).
        timeout_threshold:  Consecutive timeouts before DOWN  (default from config).
        recovery_threshold: Consecutive successes before UP   (default from config).

    Initialization behaviour:
        ``r_hat`` and ``d_hat`` are both initialised to **0.0**.
        The initial RTO is therefore 0.0.  The very first numeric
        sample is always accepted as a success (bypasses the RTO check)
        and updates the EWMA via the standard equations.  This
        deterministic initialisation is documented and tested.
    """

    def __init__(
        self,
        *,
        alpha: float | None = None,
        beta: float | None = None,
        timeout_threshold: int | None = None,
        recovery_threshold: int | None = None,
    ) -> None:
        cfg = _load_defaults()

        self._alpha = cfg["alpha"] if alpha is None else float(alpha)
        self._beta = cfg["beta"] if beta is None else float(beta)
        self._timeout_threshold = (
            cfg["timeout_threshold"] if timeout_threshold is None else int(timeout_threshold)
        )
        self._recovery_threshold = (
            cfg["recovery_threshold"] if recovery_threshold is None else int(recovery_threshold)
        )

        # EWMA state — initialised to 0, consistent with the equations
        self._r_hat: float = 0.0
        self._d_hat: float = 0.0
        self._first_sample: bool = True

        # FSM
        self._state: str = "UP"
        self._consecutive_timeouts: int = 0
        self._consecutive_successes: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current link state: ``"UP"`` or ``"DOWN"``."""
        return self._state

    @property
    def rto_ms(self) -> float:
        """Current retransmission timeout in milliseconds.

        Before the first successful sample this is **0.0** (the
        mathematical consequence of r_hat = d_hat = 0).
        """
        return self._r_hat + 4.0 * self._d_hat

    @property
    def r_hat_ms(self) -> float:
        """Current EWMA RTT estimate in milliseconds."""
        return self._r_hat

    @property
    def d_hat_ms(self) -> float:
        """Current EWMA deviation estimate in milliseconds."""
        return self._d_hat

    @property
    def consecutive_timeouts(self) -> int:
        """Number of consecutive timeouts (resets on success)."""
        return self._consecutive_timeouts

    @property
    def consecutive_successes(self) -> int:
        """Number of consecutive successes (resets on timeout)."""
        return self._consecutive_successes

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def record_sample(self, rtt_ms: float | None) -> str:
        """Record an RTT sample and update EWMA + FSM state.

        Args:
            rtt_ms:  Round-trip time in milliseconds, or ``None`` to
                     represent an explicit timeout.

        Returns:
            The link state after processing (``"UP"`` or ``"DOWN"``).

        Behaviour:
            * ``None`` is treated as a timeout.
            * A numeric value must be finite and non-negative.
            * Before the first successful sample the RTO check is
              bypassed (RTO is 0.0 and any positive sample is accepted).
            * A numeric RTT **greater than the current RTO** is treated
              as a timeout (the late/too-slow response does not feed
              into the EWMA).
            * A numeric RTT **at or below the current RTO** is treated
              as a successful sample and updates the EWMA estimates.

        Raises:
            TypeError:  If ``rtt_ms`` is not a ``float``, ``int``, or ``None``.
            ValueError: If ``rtt_ms`` is not finite or is negative.
        """
        # ---- input validation ----
        if rtt_ms is not None:
            if not isinstance(rtt_ms, (int, float)):
                raise TypeError(f"rtt_ms must be a number or None, got {type(rtt_ms).__name__}")
            rtt_ms = float(rtt_ms)
            if not math.isfinite(rtt_ms):
                raise ValueError(f"rtt_ms must be finite, got {rtt_ms}")
            if rtt_ms < 0.0:
                raise ValueError(f"rtt_ms must be non-negative, got {rtt_ms}")

        # ---- classify sample ----
        is_success = self._classify(rtt_ms)

        if is_success:
            self._handle_success(rtt_ms)  # type: ignore[arg-type]
        else:
            self._handle_timeout()

        return self._state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _classify(self, rtt_ms: float | None) -> bool:
        """Return True if the sample is a success, False for timeout."""
        if rtt_ms is None:
            return False
        if self._first_sample:
            # First numeric sample is always accepted (RTO is 0.0).
            return True
        return rtt_ms <= self.rto_ms

    def _handle_success(self, rtt_ms: float) -> None:
        """Process a successful RTT sample."""
        # EWMA update — equations from §3.2
        if self._first_sample:
            # First sample: use standard equations with r_hat_prev=0, d_hat_prev=0.
            # r_hat = alpha * r + (1-alpha) * 0 = alpha * r
            # d_hat = beta * |r - r_hat| + (1-beta) * 0
            self._r_hat = self._alpha * rtt_ms
            self._d_hat = self._beta * abs(rtt_ms - self._r_hat)
            self._first_sample = False
        else:
            # Standard EWMA update
            r_hat_prev = self._r_hat
            self._r_hat = self._alpha * rtt_ms + (1.0 - self._alpha) * r_hat_prev
            self._d_hat = self._beta * abs(rtt_ms - self._r_hat) + (1.0 - self._beta) * self._d_hat

        # FSM: success resets timeout counter
        self._consecutive_successes += 1
        self._consecutive_timeouts = 0

        if self._state == "DOWN" and self._consecutive_successes >= self._recovery_threshold:
            self._state = "UP"

    def _handle_timeout(self) -> None:
        """Process a timeout."""
        # EWMA is NOT updated — timeouts are not fed into the estimator
        self._consecutive_timeouts += 1
        self._consecutive_successes = 0

        if self._state == "UP" and self._consecutive_timeouts >= self._timeout_threshold:
            self._state = "DOWN"
