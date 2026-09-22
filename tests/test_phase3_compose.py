"""Phase 3a Step 2 — Docker Compose validation tests.

Static tests validate the Compose definition file.  Dynamic tests
(start/stop/connectivity) require Docker and docker-compose.

Run:
    pytest tests/test_phase3_compose.py -v
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_COMPOSE_FILE = Path(__file__).resolve().parent.parent / "docker-compose.yml"
_PROJECT_ROOT = _COMPOSE_FILE.parent
_IMAGE_TAG = "kinbridge-sync:phase3a"

_EXPECTED_SERVICES = {"client", "edge-a", "edge-b", "redis-a", "redis-b", "replicator"}
_EDGE_SERVICES = {"edge-a", "edge-b"}
_APP_SERVICES = {"client", "edge-a", "edge-b", "replicator"}
_REDIS_SERVICES = {"redis-a", "redis-b"}

_NETWORK_NAME = "kinbridge-net"


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def _compose_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "compose", "version"], capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


DOCKER_AVAILABLE = _docker_available()
COMPOSE_AVAILABLE = DOCKER_AVAILABLE and _compose_available()


def _run_compose(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run docker compose with project directory."""
    return subprocess.run(
        ["docker", "compose"] + list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_PROJECT_ROOT),
    )


def _load_compose() -> dict:
    """Parse docker-compose.yml."""
    with open(_COMPOSE_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Static validation (always runs)
# ---------------------------------------------------------------------------

class TestStaticComposeDefinition:
    """Validate docker-compose.yml structure without Docker."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.compose = _load_compose()

    def test_file_exists(self):
        assert _COMPOSE_FILE.exists()

    def test_has_services_key(self):
        assert "services" in self.compose

    def test_exactly_six_services(self):
        services = set(self.compose["services"].keys())
        assert services == _EXPECTED_SERVICES

    def test_all_services_use_valid_image(self):
        for name, svc in self.compose["services"].items():
            image = svc.get("image", "")
            if name in _REDIS_SERVICES:
                assert "redis" in image.lower(), (
                    f"Service {name} uses image {image}, expected redis image"
                )
            else:
                assert image == _IMAGE_TAG, (
                    f"Service {name} uses image {image}, expected {_IMAGE_TAG}"
                )

    def test_edge_a_has_net_admin(self):
        caps = self.compose["services"]["edge-a"].get("cap_add", [])
        assert "NET_ADMIN" in caps

    def test_edge_a_has_net_raw(self):
        caps = self.compose["services"]["edge-a"].get("cap_add", [])
        assert "NET_RAW" in caps

    def test_edge_b_has_net_admin(self):
        caps = self.compose["services"]["edge-b"].get("cap_add", [])
        assert "NET_ADMIN" in caps

    def test_edge_b_has_net_raw(self):
        caps = self.compose["services"]["edge-b"].get("cap_add", [])
        assert "NET_RAW" in caps

    def test_no_privileged(self):
        for name, svc in self.compose["services"].items():
            assert not svc.get("privileged", False), (
                f"Service {name} has privileged=true"
            )

    def test_redis_services_present(self):
        services = set(self.compose["services"].keys())
        assert "redis-a" in services
        assert "redis-b" in services

    def test_replicator_service_present(self):
        services = set(self.compose["services"].keys())
        assert "replicator" in services

    def test_edge_a_port_mapping(self):
        ports = self.compose["services"]["edge-a"].get("ports", [])
        assert "8000:8000" in ports

    def test_edge_b_port_mapping(self):
        ports = self.compose["services"]["edge-b"].get("ports", [])
        assert "8001:8001" in ports

    def test_has_bridge_network(self):
        networks = self.compose.get("networks", {})
        assert _NETWORK_NAME in networks
        assert networks[_NETWORK_NAME].get("driver") == "bridge"

    def test_all_services_on_bridge_network(self):
        for name, svc in self.compose["services"].items():
            svc_networks = svc.get("networks", [])
            assert _NETWORK_NAME in svc_networks, (
                f"Service {name} not on {_NETWORK_NAME}"
            )

    def test_edge_a_command_starts_uvicorn(self):
        cmd = self.compose["services"]["edge-a"].get("command", "")
        assert "uvicorn" in cmd
        assert "tool_world.main:app" in cmd
        assert "8000" in cmd

    def test_edge_a_command_starts_watcher(self):
        cmd = self.compose["services"]["edge-a"].get("command", "")
        assert "sqlite_watcher" in cmd

    def test_edge_b_command_starts_uvicorn(self):
        cmd = self.compose["services"]["edge-b"].get("command", "")
        assert "uvicorn" in cmd
        assert "tool_world.main:app" in cmd
        assert "8001" in cmd

    def test_edge_b_command_starts_writer(self):
        cmd = self.compose["services"]["edge-b"].get("command", "")
        assert "sqlite_writer" in cmd

    def test_edge_a_env_sets_port(self):
        env = self.compose["services"]["edge-a"].get("environment", [])
        env_str = " ".join(env) if isinstance(env, list) else str(env)
        assert "KINBRIDGE_PORT=8000" in env_str

    def test_edge_b_env_sets_port(self):
        env = self.compose["services"]["edge-b"].get("environment", [])
        env_str = " ".join(env) if isinstance(env, list) else str(env)
        assert "KINBRIDGE_PORT=8001" in env_str

    def test_edge_a_healthcheck_uses_health_endpoint(self):
        hc = self.compose["services"]["edge-a"].get("healthcheck", {})
        test_cmd = hc.get("test", [])
        assert any("/health" in str(c) for c in test_cmd)

    def test_edge_b_healthcheck_uses_health_endpoint(self):
        hc = self.compose["services"]["edge-b"].get("healthcheck", {})
        test_cmd = hc.get("test", [])
        assert any("/health" in str(c) for c in test_cmd)

    def test_client_has_healthcheck(self):
        hc = self.compose["services"]["client"].get("healthcheck", {})
        assert "test" in hc

    def test_redis_a_has_healthcheck(self):
        hc = self.compose["services"]["redis-a"].get("healthcheck", {})
        assert "test" in hc

    def test_redis_b_has_healthcheck(self):
        hc = self.compose["services"]["redis-b"].get("healthcheck", {})
        assert "test" in hc

    def test_replicator_command(self):
        cmd = self.compose["services"]["replicator"].get("command", "")
        assert "replicator" in cmd
        assert "--lag-ms" in cmd

    def test_edge_a_depends_on_redis_a(self):
        deps = self.compose["services"]["edge-a"].get("depends_on", {})
        assert "redis-a" in deps

    def test_edge_b_depends_on_redis_b(self):
        deps = self.compose["services"]["edge-b"].get("depends_on", {})
        assert "redis-b" in deps

    def test_replicator_depends_on_both_redis(self):
        deps = self.compose["services"]["replicator"].get("depends_on", {})
        assert "redis-a" in deps
        assert "redis-b" in deps

    def test_redis_a_port_mapping(self):
        ports = self.compose["services"]["redis-a"].get("ports", [])
        assert "6379:6379" in ports

    def test_redis_b_port_mapping(self):
        ports = self.compose["services"]["redis-b"].get("ports", [])
        assert "6380:6379" in ports

    def test_edge_a_env_sets_redis_url(self):
        env = self.compose["services"]["edge-a"].get("environment", [])
        env_str = " ".join(env) if isinstance(env, list) else str(env)
        assert "REDIS_A_URL" in env_str

    def test_edge_b_env_sets_redis_url(self):
        env = self.compose["services"]["edge-b"].get("environment", [])
        env_str = " ".join(env) if isinstance(env, list) else str(env)
        assert "REDIS_B_URL" in env_str

    def test_volumes_defined(self):
        volumes = self.compose.get("volumes", {})
        assert "redis-a-data" in volumes
        assert "redis-b-data" in volumes
        assert "edge-a-data" in volumes
        assert "edge-b-data" in volumes


# ---------------------------------------------------------------------------
# Compose CLI validation (requires Docker)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not COMPOSE_AVAILABLE, reason="docker compose not available")
class TestComposeCLI:
    """Validate compose config and stack lifecycle via docker compose CLI."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Ensure stack is down after each test."""
        yield
        _run_compose("down", "-v", "--remove-orphans", timeout=60)

    def test_compose_config_parses(self):
        result = _run_compose("config")
        assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"

    def test_compose_config_services(self):
        result = _run_compose("config", "--format", "json")
        assert result.returncode == 0
        config = json.loads(result.stdout)
        services = set(config.get("services", {}).keys())
        assert services == _EXPECTED_SERVICES

    def test_compose_up_and_down(self):
        result = _run_compose("up", "-d", "--wait", timeout=180)
        assert result.returncode == 0, f"docker compose up failed:\n{result.stderr}"

        result = _run_compose("ps", "--format", "json")
        assert result.returncode == 0
        lines = [l for l in result.stdout.strip().splitlines() if l]
        running = 0
        for line in lines:
            svc = json.loads(line)
            if svc.get("State") == "running":
                running += 1
        assert running == 3, f"Expected 3 running services, got {running}"

    def test_edge_a_health(self):
        result = _run_compose("up", "-d", "--wait", timeout=180)
        assert result.returncode == 0

        # Wait for health check
        for _ in range(30):
            r = subprocess.run(
                ["docker", "exec", "kb-edge-a", "python", "-c",
                 "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"],
                capture_output=True, timeout=10,
            )
            if r.returncode == 0:
                break
            time.sleep(2)
        else:
            pytest.fail("edge-a /health not ready after 60s")

    def test_edge_b_health(self):
        result = _run_compose("up", "-d", "--wait", timeout=180)
        assert result.returncode == 0

        for _ in range(30):
            r = subprocess.run(
                ["docker", "exec", "kb-edge-b", "python", "-c",
                 "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"],
                capture_output=True, timeout=10,
            )
            if r.returncode == 0:
                break
            time.sleep(2)
        else:
            pytest.fail("edge-b /health not ready after 60s")


# ---------------------------------------------------------------------------
# Connectivity tests (requires running stack)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not COMPOSE_AVAILABLE, reason="docker compose not available")
class TestConnectivity:
    """Verify inter-container connectivity by service name."""

    @pytest.fixture(autouse=True)
    def _stack(self):
        """Start and stop the compose stack."""
        result = _run_compose("up", "-d", "--wait", timeout=180)
        if result.returncode != 0:
            pytest.skip(f"Could not start stack: {result.stderr}")
        # Allow containers to settle
        time.sleep(5)
        yield
        _run_compose("down", "-v", "--remove-orphans", timeout=60)

    def test_client_can_resolve_edge_a(self):
        result = subprocess.run(
            ["docker", "exec", "kb-client", "python", "-c",
             "import socket; print(socket.gethostbyname('edge-a'))"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"DNS resolution failed: {result.stderr}"
        assert result.stdout.strip(), "edge-a resolved to empty"

    def test_client_can_resolve_edge_b(self):
        result = subprocess.run(
            ["docker", "exec", "kb-client", "python", "-c",
             "import socket; print(socket.gethostbyname('edge-b'))"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"DNS resolution failed: {result.stderr}"
        assert result.stdout.strip(), "edge-b resolved to empty"

    def test_client_can_reach_edge_a_health(self):
        result = subprocess.run(
            ["docker", "exec", "kb-client", "python", "-c",
             "import urllib.request; r = urllib.request.urlopen('http://edge-a:8000/health'); print(r.read().decode())"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, f"HTTP request failed: {result.stderr}"
        assert "ok" in result.stdout

    def test_client_can_reach_edge_b_health(self):
        result = subprocess.run(
            ["docker", "exec", "kb-client", "python", "-c",
             "import urllib.request; r = urllib.request.urlopen('http://edge-b:8001/health'); print(r.read().decode())"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, f"HTTP request failed: {result.stderr}"
        assert "ok" in result.stdout

    def test_edge_a_can_resolve_edge_b(self):
        result = subprocess.run(
            ["docker", "exec", "kb-edge-a", "python", "-c",
             "import socket; print(socket.gethostbyname('edge-b'))"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert result.stdout.strip()

    def test_edge_b_can_resolve_edge_a(self):
        result = subprocess.run(
            ["docker", "exec", "kb-edge-b", "python", "-c",
             "import socket; print(socket.gethostbyname('edge-a'))"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert result.stdout.strip()

    def test_edge_a_can_resolve_redis_a(self):
        result = subprocess.run(
            ["docker", "exec", "kb-edge-a", "python", "-c",
             "import socket; print(socket.gethostbyname('redis-a'))"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert result.stdout.strip()

    def test_edge_b_can_resolve_redis_b(self):
        result = subprocess.run(
            ["docker", "exec", "kb-edge-b", "python", "-c",
             "import socket; print(socket.gethostbyname('redis-b'))"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert result.stdout.strip()

    def test_replicator_can_resolve_both_redis(self):
        result = subprocess.run(
            ["docker", "exec", "kb-replicator", "python", "-c",
             "import socket; print(socket.gethostbyname('redis-a')); print(socket.gethostbyname('redis-b'))"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 2
