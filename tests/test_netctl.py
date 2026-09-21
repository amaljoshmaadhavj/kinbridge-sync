"""Phase 3a Step 3 — netctl.py validation tests.

Unit tests validate command construction and input validation without
modifying any real network state.  Integration tests run against the
Docker Compose stack to verify end-to-end network control.

Run unit tests only (no Docker):
    pytest tests/test_netctl.py -v -k "not Integration"

Run all tests (requires Docker + Compose stack):
    pytest tests/test_netctl.py -v
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure experiments/ is importable
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.netctl import (
    INTERFACE,
    SUPPORTED_TARGETS,
    apply_blackout,
    apply_delay,
    apply_loss,
    apply_restore,
    build_iptables_command,
    build_tc_command,
    get_container_id,
    main,
    parse_args,
    run_docker_exec,
    validate_delay,
    validate_loss,
    validate_target,
)


# ─────────────────────────────────────────────────────────────
# CLI parsing tests
# ─────────────────────────────────────────────────────────────

class TestCLIParsing:
    """Validate CLI argument parsing."""

    def test_delay_parsing(self):
        args = parse_args(["delay", "--target", "client", "--ms", "100"])
        assert args.operation == "delay"
        assert args.target == "client"
        assert args.ms == "100"

    def test_loss_parsing(self):
        args = parse_args(["loss", "--target", "edge-a", "--percent", "5"])
        assert args.operation == "loss"
        assert args.target == "edge-a"
        assert args.percent == "5"

    def test_blackout_parsing(self):
        args = parse_args(["blackout", "--target", "edge-b"])
        assert args.operation == "blackout"
        assert args.target == "edge-b"

    def test_restore_parsing(self):
        args = parse_args(["restore", "--target", "client"])
        assert args.operation == "restore"
        assert args.target == "client"

    def test_missing_operation_fails(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_missing_target_fails(self):
        with pytest.raises(SystemExit):
            parse_args(["delay", "--ms", "100"])

    def test_missing_ms_fails(self):
        with pytest.raises(SystemExit):
            parse_args(["delay", "--target", "client"])

    def test_missing_percent_fails(self):
        with pytest.raises(SystemExit):
            parse_args(["loss", "--target", "client"])


# ─────────────────────────────────────────────────────────────
# Target validation tests
# ─────────────────────────────────────────────────────────────

class TestTargetValidation:
    """Validate target name validation."""

    def test_valid_targets(self):
        for target in SUPPORTED_TARGETS:
            assert validate_target(target) == target

    def test_invalid_target(self):
        with pytest.raises(ValueError, match="Invalid target"):
            validate_target("nonexistent")

    def test_invalid_target_empty(self):
        with pytest.raises(ValueError, match="Invalid target"):
            validate_target("")

    def test_invalid_target_case(self):
        with pytest.raises(ValueError, match="Invalid target"):
            validate_target("Client")

    def test_invalid_target_with_docker_prefix(self):
        with pytest.raises(ValueError, match="Invalid target"):
            validate_target("kb-edge-a")


# ─────────────────────────────────────────────────────────────
# Delay validation tests
# ─────────────────────────────────────────────────────────────

class TestDelayValidation:
    """Validate delay value parsing."""

    def test_valid_delay_zero(self):
        assert validate_delay("0") == 0

    def test_valid_delay_positive(self):
        assert validate_delay("100") == 100

    def test_valid_delay_large(self):
        assert validate_delay("5000") == 5000

    def test_invalid_delay_string(self):
        with pytest.raises(ValueError, match="Invalid delay"):
            validate_delay("abc")

    def test_invalid_delay_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_delay("-1")

    def test_invalid_delay_float(self):
        with pytest.raises(ValueError, match="Invalid delay"):
            validate_delay("1.5")

    def test_invalid_delay_empty(self):
        with pytest.raises(ValueError, match="Invalid delay"):
            validate_delay("")


# ─────────────────────────────────────────────────────────────
# Loss validation tests
# ─────────────────────────────────────────────────────────────

class TestLossValidation:
    """Validate loss percentage parsing."""

    def test_valid_loss_zero(self):
        assert validate_loss("0") == 0.0

    def test_valid_loss_small(self):
        assert validate_loss("0.5") == 0.5

    def test_valid_loss_50(self):
        assert validate_loss("50") == 50.0

    def test_valid_loss_100(self):
        assert validate_loss("100") == 100.0

    def test_invalid_loss_string(self):
        with pytest.raises(ValueError, match="Invalid loss"):
            validate_loss("abc")

    def test_invalid_loss_negative(self):
        with pytest.raises(ValueError, match="(?i)must be in range"):
            validate_loss("-1")

    def test_invalid_loss_over_100(self):
        with pytest.raises(ValueError, match="(?i)must be in range"):
            validate_loss("101")

    def test_invalid_loss_empty(self):
        with pytest.raises(ValueError, match="Invalid loss"):
            validate_loss("")


# ─────────────────────────────────────────────────────────────
# Unsupported operation tests
# ─────────────────────────────────────────────────────────────

class TestUnsupportedOperations:
    """Validate rejection of unsupported operations."""

    def test_unknown_operation(self):
        with pytest.raises(SystemExit):
            main(["bogus", "--target", "client"])

    def test_invalid_iptables_operation(self):
        with pytest.raises(ValueError, match="Unsupported iptables operation"):
            build_iptables_command("bogus")


# ─────────────────────────────────────────────────────────────
# Command construction tests (no network changes)
# ─────────────────────────────────────────────────────────────

class TestCommandConstruction:
    """Validate Docker/tc/iptables command construction."""

    def test_tc_delay_command(self):
        cmd = build_tc_command(delay_ms=100)
        assert cmd == ["tc", "qdisc", "replace", "dev", INTERFACE,
                       "root", "netem", "delay", "100ms"]

    def test_tc_loss_command(self):
        cmd = build_tc_command(loss_pct=5.0)
        assert cmd == ["tc", "qdisc", "replace", "dev", INTERFACE,
                       "root", "netem", "loss", "5.0%"]

    def test_tc_delay_zero(self):
        cmd = build_tc_command(delay_ms=0)
        assert "0ms" in cmd

    def test_tc_loss_zero(self):
        cmd = build_tc_command(loss_pct=0.0)
        assert "0.0%" in cmd

    def test_tc_both_raises(self):
        with pytest.raises(ValueError, match="not both"):
            build_tc_command(delay_ms=100, loss_pct=5.0)

    def test_tc_neither_raises(self):
        with pytest.raises(ValueError, match="Specify either"):
            build_tc_command()

    def test_iptables_blackout_command(self):
        cmd = build_iptables_command("blackout")
        assert cmd == ["iptables", "-I", "OUTPUT", "-j", "DROP"]

    def test_iptables_restore_command(self):
        cmd = build_iptables_command("restore")
        assert cmd == ["iptables", "-F"]

    def test_docker_exec_prefix(self):
        with patch("experiments.netctl.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_docker_exec("my-container", ["echo", "hello"])
            mock_run.assert_called_once_with(
                ["docker", "exec", "my-container", "echo", "hello"],
                capture_output=True,
                text=True,
            )


# ─────────────────────────────────────────────────────────────
# Subprocess failure / non-zero exit tests
# ─────────────────────────────────────────────────────────────

class TestSubprocessFailure:
    """Validate error handling when subprocess commands fail."""

    def test_delay_docker_exec_failure(self):
        with patch("experiments.netctl.run_docker_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("docker not found")
            rc = apply_delay("edge-a", 100)
            assert rc == 1

    def test_delay_tc_command_fails(self):
        with patch("experiments.netctl.run_docker_exec") as mock_exec:
            mock_exec.return_value = MagicMock(
                returncode=2, stdout="", stderr="RTNETLINK error"
            )
            rc = apply_delay("edge-a", 100)
            assert rc == 2

    def test_loss_docker_exec_failure(self):
        with patch("experiments.netctl.run_docker_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("docker not found")
            rc = apply_loss("edge-a", 5.0)
            assert rc == 1

    def test_blackout_docker_exec_failure(self):
        with patch("experiments.netctl.run_docker_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("docker not found")
            rc = apply_blackout("edge-a")
            assert rc == 1

    def test_restore_docker_exec_failure(self):
        with patch("experiments.netctl.run_docker_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("docker not found")
            rc = apply_restore("edge-a")
            assert rc == 1

    def test_restore_iptables_fails_tc_still_attempted(self):
        with patch("experiments.netctl.run_docker_exec") as mock_exec:
            # First call (iptables) fails, second call (tc del) succeeds
            mock_exec.side_effect = [
                MagicMock(returncode=1, stdout="", stderr="iptables error"),
                MagicMock(returncode=0, stdout="", stderr=""),
            ]
            rc = apply_restore("edge-a")
            assert rc == 1
            assert mock_exec.call_count == 2

    def test_invalid_target_returns_1(self):
        rc = main(["delay", "--target", "bogus", "--ms", "100"])
        assert rc == 1

    def test_invalid_delay_returns_1(self):
        rc = main(["delay", "--target", "client", "--ms", "abc"])
        assert rc == 1

    def test_invalid_loss_returns_1(self):
        rc = main(["loss", "--target", "client", "--percent", "999"])
        assert rc == 1


# ─────────────────────────────────────────────────────────────
# Shell interpolation protection tests
# ─────────────────────────────────────────────────────────────

class TestShellInterpolationProtection:
    """Ensure commands are not vulnerable to shell injection."""

    def test_docker_exec_uses_list_not_shell(self):
        with patch("experiments.netctl.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_docker_exec("container", ["cmd", "arg1"])
            # Verify subprocess.run was called with a list, not a string
            call_args = mock_run.call_args
            assert isinstance(call_args[0][0], list)

    def test_tc_command_uses_list(self):
        cmd = build_tc_command(delay_ms=100)
        assert isinstance(cmd, list)
        assert all(isinstance(part, str) for part in cmd)

    def test_iptables_command_uses_list(self):
        cmd = build_iptables_command("blackout")
        assert isinstance(cmd, list)
        assert all(isinstance(part, str) for part in cmd)


# ─────────────────────────────────────────────────────────────
# Container resolution tests
# ─────────────────────────────────────────────────────────────

class TestContainerResolution:
    """Validate container ID resolution logic."""

    def test_compose_ps_success(self):
        with patch("experiments.netctl.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="abc123\n", stderr=""
            )
            cid = get_container_id("edge-a")
            assert cid == "abc123"

    def test_compose_ps_empty_falls_back_to_label(self):
        with patch("experiments.netctl.subprocess.run") as mock_run:
            # First call (compose ps) returns empty
            # Second call (docker ps label) returns container
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="", stderr=""),
                MagicMock(returncode=0, stdout="def456\n", stderr=""),
            ]
            cid = get_container_id("edge-a")
            assert cid == "def456"

    def test_all_fallbacks_empty_returns_none(self):
        with patch("experiments.netctl.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            cid = get_container_id("nonexistent")
            assert cid is None

    def test_compose_ps_timeout_falls_back(self):
        with patch("experiments.netctl.subprocess.run") as mock_run:
            mock_run.side_effect = [
                subprocess.TimeoutExpired(cmd="docker", timeout=10),
                MagicMock(returncode=0, stdout="ghi789\n", stderr=""),
            ]
            cid = get_container_id("edge-b")
            assert cid == "ghi789"


# ─────────────────────────────────────────────────────────────
# Docker integration tests (requires running Compose stack)
# ─────────────────────────────────────────────────────────────

def _compose_running() -> bool:
    """Check if the Compose stack is up."""
    import os
    os.chdir(str(_PROJECT_ROOT))
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "-q", "edge-a"],
            capture_output=True, text=True, timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


COMPOSE_RUNNING = _compose_running()


@pytest.mark.skipif(not COMPOSE_RUNNING,
                    reason="Compose stack not running")
class TestIntegration:
    """Integration tests against a running Compose stack."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, request):
        """Restore network state after each test."""
        yield
        # Always restore after each test
        for svc in ("edge-a", "edge-b"):
            apply_restore(svc)

    def _exec(self, service: str, cmd: list) -> subprocess.CompletedProcess:
        """Execute a command in the resolved container for a service."""
        cid = get_container_id(service) or service
        return run_docker_exec(cid, cmd)

    def test_delay_applied(self):
        rc = apply_delay("edge-a", 50)
        assert rc == 0

        # Verify tc qdisc shows netem delay
        result = self._exec("edge-a", ["tc", "qdisc", "show"])
        assert result.returncode == 0
        assert "netem" in result.stdout
        assert "50ms" in result.stdout

    def test_loss_applied(self):
        rc = apply_loss("edge-a", 10.0)
        assert rc == 0

        result = self._exec("edge-a", ["tc", "qdisc", "show"])
        assert result.returncode == 0
        assert "netem" in result.stdout
        assert "10%" in result.stdout

    def test_blackout_applied(self):
        rc = apply_blackout("edge-a")
        assert rc == 0

        result = self._exec("edge-a", ["iptables", "-L", "OUTPUT", "-n"])
        assert result.returncode == 0
        assert "DROP" in result.stdout

    def test_restore_removes_tc(self):
        apply_delay("edge-a", 100)
        rc = apply_restore("edge-a")
        assert rc == 0

        result = self._exec("edge-a", ["tc", "qdisc", "show"])
        assert result.returncode == 0
        assert "netem" not in result.stdout

    def test_restore_removes_iptables(self):
        apply_blackout("edge-a")
        rc = apply_restore("edge-a")
        assert rc == 0

        result = self._exec("edge-a", ["iptables", "-L", "OUTPUT", "-n"])
        assert result.returncode == 0
        assert "DROP" not in result.stdout

    def test_delay_replace_existing(self):
        """Apply delay twice — should replace, not fail."""
        rc1 = apply_delay("edge-a", 50)
        assert rc1 == 0
        rc2 = apply_delay("edge-a", 200)
        assert rc2 == 0

        result = self._exec("edge-a", ["tc", "qdisc", "show"])
        assert "200ms" in result.stdout
        assert "50ms" not in result.stdout

    def test_restore_on_clean_container(self):
        """Restore on a container with no tc/iptables — should not fail."""
        rc = apply_restore("edge-a")
        assert rc == 0

    def test_target_resolution(self):
        cid = get_container_id("edge-a")
        assert cid is not None
        assert len(cid) > 0
