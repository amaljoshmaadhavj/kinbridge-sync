"""Phase 3a Step 1 — Container image validation tests.

These tests verify the Docker image definition is correct.  When Docker
is available, they build the image and verify runtime properties.  When
Docker is not available, they perform static validation only.

Run:
    pytest tests/test_docker_image.py -v
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_IMAGE_TAG = "kinbridge-sync:phase3a"
_DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"
_PROJECT_ROOT = _DOCKERFILE.parent

# Required system tools inside the container
_SYSTEM_TOOLS = ["tc", "iptables"]

# Required Python packages inside the container
_PYTHON_IMPORTS = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("yaml", "yaml"),
]

# Source packages that must be importable inside the container
_SOURCE_IMPORTS = [
    ("tool_world", "tool_world"),
    ("tool_world.config", "tool_world.config"),
    ("tool_world.models", "tool_world.models"),
    ("tool_world.ledger", "tool_world.ledger"),
    ("tool_world.main", "tool_world.main"),
    ("client", "client"),
    ("client.buffer", "client.buffer"),
    ("client.link_monitor", "client.link_monitor"),
]


def _docker_available() -> bool:
    """Return True if the Docker daemon is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


DOCKER_AVAILABLE = _docker_available()


# ---------------------------------------------------------------------------
# Static validation (always runs)
# ---------------------------------------------------------------------------

class TestStaticValidation:
    """Validate the Dockerfile and source layout without Docker."""

    def test_dockerfile_exists(self):
        assert _DOCKERFILE.exists(), f"Dockerfile not found at {_DOCKERFILE}"

    def test_dockerfile_references_base_image(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "python:3.12-slim" in content, "Base image must be python:3.12-slim"

    def test_dockerfile_installs_iproute2(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "iproute2" in content, "Must install iproute2 (tc command)"

    def test_dockerfile_installs_iptables(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "iptables" in content, "Must install iptables"

    def test_dockerfile_installs_fastapi(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "fastapi" in content, "Must install fastapi"

    def test_dockerfile_installs_uvicon(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "uvicorn" in content, "Must install uvicorn"

    def test_dockerfile_installs_pydantic(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "pydantic" in content, "Must install pydantic"

    def test_dockerfile_installs_pyyaml(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "pyyaml" in content, "Must install pyyaml"

    def test_dockerfile_sets_pythonpath(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "PYTHONPATH=/app" in content, "Must set PYTHONPATH=/app"

    def test_dockerfile_copies_tool_world(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "COPY tool_world/" in content, "Must copy tool_world/"

    def test_dockerfile_copies_client(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "COPY client/" in content, "Must copy client/"

    def test_dockerfile_copies_pilot_tools(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "COPY pilot/tools.py" in content, "Must copy pilot/tools.py"

    def test_source_files_exist(self):
        """All files referenced by COPY instructions must exist."""
        required = [
            "tool_world/__init__.py",
            "tool_world/main.py",
            "tool_world/config.py",
            "tool_world/ledger.py",
            "tool_world/models.py",
            "client/__init__.py",
            "client/buffer.py",
            "client/link_monitor.py",
            "pilot/tools.py",
            "pilot/__init__.py",
            "core/__init__.py",
            "core/config.py",
        ]
        for rel_path in required:
            full = _PROJECT_ROOT / rel_path
            assert full.exists(), f"Required source file missing: {rel_path}"

    def test_no_privileged_in_dockerfile(self):
        """Dockerfile must not add privileged mode."""
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "--privileged" not in content.lower(), \
            "Dockerfile must not use --privileged"

    def test_no_net_admin_in_dockerfile(self):
        """Capabilities are supplied by docker-compose, not Dockerfile."""
        content = _DOCKERFILE.read_text(encoding="utf-8")
        # Check non-comment lines only
        non_comment_lines = [
            line for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        non_comment_text = "\n".join(non_comment_lines)
        assert "NET_ADMIN" not in non_comment_text, \
            "NET_ADMIN capability must be in docker-compose, not Dockerfile"

    def test_python_imports_locally(self):
        """All required packages must be importable on the host."""
        for module_name, _ in _PYTHON_IMPORTS:
            __import__(module_name)

    def test_source_imports_locally(self):
        """Source packages must be importable on the host."""
        old_path = sys.path[:]
        try:
            sys.path.insert(0, str(_PROJECT_ROOT))
            for module_name, _ in _SOURCE_IMPORTS:
                __import__(module_name)
        finally:
            sys.path[:] = old_path


# ---------------------------------------------------------------------------
# Runtime validation (requires Docker)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker daemon not available")
class TestImageBuild:
    """Build the image and verify runtime properties."""

    @pytest.fixture(autouse=True)
    def _build_image(self):
        """Build the image once for all tests in this class."""
        result = subprocess.run(
            ["docker", "build", "-t", _IMAGE_TAG, "."],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"Docker build failed:\n{result.stderr}"
        yield

    def _run_in_container(self, command: list[str]) -> subprocess.CompletedProcess:
        """Run a command inside the built image."""
        return subprocess.run(
            ["docker", "run", "--rm", _IMAGE_TAG] + command,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_python_exists(self):
        result = self._run_in_container(["python", "--version"])
        assert result.returncode == 0
        assert "Python 3.12" in result.stdout

    def test_tc_exists(self):
        result = self._run_in_container(["which", "tc"])
        assert result.returncode == 0, "tc (iproute2) not found in image"

    def test_iptables_exists(self):
        result = self._run_in_container(["which", "iptables"])
        assert result.returncode == 0, "iptables not found in image"

    def test_import_tool_world(self):
        result = self._run_in_container([
            "python", "-c",
            "from tool_world.main import app; print('tool_world.main: OK')"
        ])
        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert "OK" in result.stdout

    def test_import_client_buffer(self):
        result = self._run_in_container([
            "python", "-c",
            "from client.buffer import ActionBuffer; print('buffer: OK')"
        ])
        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert "OK" in result.stdout

    def test_import_client_link_monitor(self):
        result = self._run_in_container([
            "python", "-c",
            "from client.link_monitor import LinkMonitor; print('link_monitor: OK')"
        ])
        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert "OK" in result.stdout

    def test_import_fastapi(self):
        result = self._run_in_container([
            "python", "-c", "import fastapi; print('fastapi:', fastapi.__version__)"
        ])
        assert result.returncode == 0
        assert "fastapi:" in result.stdout

    def test_import_pyyaml(self):
        result = self._run_in_container([
            "python", "-c", "import yaml; print('yaml: OK')"
        ])
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_pythonpath_set(self):
        result = self._run_in_container([
            "python", "-c",
            "import os; print('PYTHONPATH:', os.environ.get('PYTHONPATH', 'NOT SET'))"
        ])
        assert result.returncode == 0
        assert "PYTHONPATH: /app" in result.stdout
