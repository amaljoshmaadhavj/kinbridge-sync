"""Docker network testbed — tc netem / iptables control (Phase 3a Step 3).

Controls network conditions inside Docker Compose containers using
Linux tc (traffic control) and iptables, executed via Docker exec.

Supported operations:
    delay   — add constant delay via tc netem
    loss    — add packet loss via tc netem
    blackout — block all outbound traffic via iptables
    restore — remove all tc netem and iptables filter rules

All commands execute inside the target container, never on the host.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Optional


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

SUPPORTED_TARGETS = frozenset({"client", "edge-a", "edge-b"})
INTERFACE = "eth0"


# ─────────────────────────────────────────────────────────────
# Docker execution
# ─────────────────────────────────────────────────────────────

def run_docker_exec(container: str, cmd: List[str]) -> subprocess.CompletedProcess:
    """Execute a command inside a Docker container via docker exec.

    Args:
        container: Container name or ID.
        cmd: Command and arguments to execute inside the container.

    Returns:
        CompletedProcess with captured output.

    Raises:
        subprocess.CalledProcessError: If the docker exec command fails.
        FileNotFoundError: If the docker CLI is not available.
    """
    full_cmd = ["docker", "exec", container] + cmd
    return subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
    )


# ─────────────────────────────────────────────────────────────
# Command construction
# ─────────────────────────────────────────────────────────────

def build_tc_command(delay_ms: Optional[int] = None,
                     loss_pct: Optional[float] = None) -> List[str]:
    """Build the tc qdisc command for netem delay or loss.

    Uses 'replace' so an existing root qdisc is overwritten cleanly
    instead of failing with 'RTNETLINK answers: File exists'.

    Args:
        delay_ms: Delay in milliseconds (for delay operation).
        loss_pct: Loss percentage (for loss operation).

    Returns:
        Command as a list of strings.

    Raises:
        ValueError: If neither delay_ms nor loss_pct is provided,
            or if both are provided.
    """
    if delay_ms is not None and loss_pct is not None:
        raise ValueError("Specify either delay_ms or loss_pct, not both")
    if delay_ms is None and loss_pct is None:
        raise ValueError("Specify either delay_ms or loss_pct")

    parts = ["tc", "qdisc", "replace", "dev", INTERFACE, "root", "netem"]
    if delay_ms is not None:
        parts.extend(["delay", f"{delay_ms}ms"])
    elif loss_pct is not None:
        parts.extend(["loss", f"{loss_pct}%"])
    return parts


def build_iptables_command(operation: str) -> List[str]:
    """Build the iptables command for blackout or restore.

    Args:
        operation: Either 'blackout' or 'restore'.

    Returns:
        Command as a list of strings.

    Raises:
        ValueError: If operation is not 'blackout' or 'restore'.
    """
    if operation == "blackout":
        return ["iptables", "-I", "OUTPUT", "-j", "DROP"]
    if operation == "restore":
        return ["iptables", "-F"]
    raise ValueError(f"Unsupported iptables operation: {operation}")


# ─────────────────────────────────────────────────────────────
# Container resolution
# ─────────────────────────────────────────────────────────────

def get_container_id(service_name: str) -> Optional[str]:
    """Resolve a Compose service name to a running container ID.

    Uses ``docker compose ps`` to find the container for the given
    service in the current project, then falls back to label-based
    lookup via ``docker ps``.

    Args:
        service_name: Compose service name (e.g. 'edge-a').

    Returns:
        Container ID string, or None if not found.
    """
    # Try docker compose ps first (works within project directory)
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "-q", service_name],
            capture_output=True, text=True, timeout=10,
        )
        cid = result.stdout.strip()
        if cid:
            return cid
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: find by Docker label
    try:
        result = subprocess.run(
            ["docker", "ps", "-q",
             "--filter", f"label=com.docker.compose.service={service_name}"],
            capture_output=True, text=True, timeout=10,
        )
        cid = result.stdout.strip()
        if cid:
            return cid.splitlines()[0]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: find by container name containing the service name
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name={service_name}"],
            capture_output=True, text=True, timeout=10,
        )
        cid = result.stdout.strip()
        if cid:
            return cid.splitlines()[0]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


# ─────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────

def validate_target(target: str) -> str:
    """Validate and normalise a target service name.

    Args:
        target: Raw target string.

    Returns:
        Normalised target name.

    Raises:
        ValueError: If target is not in SUPPORTED_TARGETS.
    """
    if target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Invalid target '{target}'. "
            f"Must be one of: {', '.join(sorted(SUPPORTED_TARGETS))}"
        )
    return target


def validate_delay(ms: str) -> int:
    """Parse and validate a delay value.

    Args:
        ms: Delay string in milliseconds.

    Returns:
        Integer delay in milliseconds.

    Raises:
        ValueError: If the value is not a valid non-negative integer.
    """
    try:
        value = int(ms)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid delay value '{ms}'. Must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"Invalid delay value '{ms}'. Must be non-negative.")
    return value


def validate_loss(percent: str) -> float:
    """Parse and validate a loss percentage.

    Args:
        percent: Loss percentage string.

    Returns:
        Float loss percentage.

    Raises:
        ValueError: If the value is not a valid number in [0, 100].
    """
    try:
        value = float(percent)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid loss value '{percent}'. Must be a number.")
    if value < 0 or value > 100:
        raise ValueError(f"Invalid loss value '{percent}'. Must be in range [0, 100].")
    return value


# ─────────────────────────────────────────────────────────────
# Operations
# ─────────────────────────────────────────────────────────────

def apply_delay(service_name: str, delay_ms: int) -> int:
    """Apply a netem delay to the target container.

    Args:
        service_name: Compose service name (e.g. 'edge-a') or container ID.
        delay_ms: Delay in milliseconds.

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    container = get_container_id(service_name) or service_name
    cmd = build_tc_command(delay_ms=delay_ms)
    try:
        result = run_docker_exec(container, cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Error executing delay: {exc}", file=sys.stderr)
        return 1
    if result.returncode != 0:
        print(f"tc command failed: {result.stderr.strip()}", file=sys.stderr)
        return result.returncode
    print(f"Delay {delay_ms}ms applied to {service_name}")
    return 0


def apply_loss(service_name: str, loss_pct: float) -> int:
    """Apply netem packet loss to the target container.

    Args:
        service_name: Compose service name (e.g. 'edge-a') or container ID.
        loss_pct: Loss percentage.

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    container = get_container_id(service_name) or service_name
    cmd = build_tc_command(loss_pct=loss_pct)
    try:
        result = run_docker_exec(container, cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Error executing loss: {exc}", file=sys.stderr)
        return 1
    if result.returncode != 0:
        print(f"tc command failed: {result.stderr.strip()}", file=sys.stderr)
        return result.returncode
    print(f"Loss {loss_pct}% applied to {service_name}")
    return 0


def apply_blackout(service_name: str) -> int:
    """Block all outbound traffic via iptables.

    Args:
        service_name: Compose service name (e.g. 'edge-a') or container ID.

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    container = get_container_id(service_name) or service_name
    cmd = build_iptables_command("blackout")
    try:
        result = run_docker_exec(container, cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Error executing blackout: {exc}", file=sys.stderr)
        return 1
    if result.returncode != 0:
        print(f"iptables command failed: {result.stderr.strip()}", file=sys.stderr)
        return result.returncode
    print(f"Blackout applied to {service_name}")
    return 0


def apply_restore(service_name: str) -> int:
    """Remove all tc netem and iptables filter rules.

    Flushes:
    - iptables OUTPUT chain filter rules
    - tc root qdisc on eth0

    Args:
        service_name: Compose service name (e.g. 'edge-a') or container ID.

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    container = get_container_id(service_name) or service_name
    exit_code = 0

    # Flush iptables filter rules
    cmd = build_iptables_command("restore")
    try:
        result = run_docker_exec(container, cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Error flushing iptables: {exc}", file=sys.stderr)
        exit_code = 1
    else:
        if result.returncode != 0:
            print(f"iptables flush failed: {result.stderr.strip()}",
                  file=sys.stderr)
            exit_code = result.returncode

    # Remove tc root qdisc (ignore errors if none exists)
    tc_cmd = ["tc", "qdisc", "del", "dev", INTERFACE, "root"]
    try:
        result = run_docker_exec(container, tc_cmd)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    if exit_code == 0:
        print(f"Restore completed for {service_name}")
    return exit_code


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="netctl",
        description="Docker network testbed control — tc netem / iptables",
    )
    sub = parser.add_subparsers(dest="operation", required=True)

    # delay
    p_delay = sub.add_parser("delay", help="Add constant delay via tc netem")
    p_delay.add_argument("--target", required=True,
                         help="Target container: client, edge-a, edge-b")
    p_delay.add_argument("--ms", required=True,
                         help="Delay in milliseconds (non-negative integer)")

    # loss
    p_loss = sub.add_parser("loss", help="Add packet loss via tc netem")
    p_loss.add_argument("--target", required=True,
                        help="Target container: client, edge-a, edge-b")
    p_loss.add_argument("--percent", required=True,
                        help="Loss percentage (0-100)")

    # blackout
    p_blackout = sub.add_parser("blackout",
                                help="Block all outbound traffic via iptables")
    p_blackout.add_argument("--target", required=True,
                            help="Target container: client, edge-a, edge-b")

    # restore
    p_restore = sub.add_parser("restore",
                               help="Remove all tc netem and iptables rules")
    p_restore.add_argument("--target", required=True,
                           help="Target container: client, edge-a, edge-b")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the netctl CLI.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 on success).
    """
    args = parse_args(argv)

    # Validate target
    try:
        target = validate_target(args.target)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Validate operation-specific arguments
    if args.operation == "delay":
        try:
            delay_ms = validate_delay(args.ms)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return apply_delay(target, delay_ms)

    if args.operation == "loss":
        try:
            loss_pct = validate_loss(args.percent)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return apply_loss(target, loss_pct)

    if args.operation == "blackout":
        return apply_blackout(target)

    if args.operation == "restore":
        return apply_restore(target)

    print(f"Error: Unknown operation '{args.operation}'", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
