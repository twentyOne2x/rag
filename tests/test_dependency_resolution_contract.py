from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HUGGINGFACE_HUB_PIN = "huggingface-hub==0.36.0"
TORCH_CPU_PIN = (
    "torch @ https://download-r2.pytorch.org/whl/cpu/"
    "torch-2.14.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"
    "#sha256=673dbf5c9bbadfffab7a386b6dd7a0c219f1408a328b7b4e86d0ae551cdafa42"
    ' ; sys_platform == "linux" and platform_machine == "x86_64"'
)


def _active_requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_huggingface_hub_resolver_boundary_is_exact_in_runtime_and_ci() -> None:
    for filename in ("requirements.txt", "requirements-ci.txt"):
        requirements = _active_requirements(ROOT / filename)
        hub_requirements = [
            requirement
            for requirement in requirements
            if requirement.lower().startswith("huggingface-hub")
        ]
        assert hub_requirements == [HUGGINGFACE_HUB_PIN], filename


def test_linux_amd64_torch_uses_hash_bound_cpu_wheel_in_runtime_and_ci() -> None:
    for filename in ("requirements.txt", "requirements-ci.txt"):
        requirements = _active_requirements(ROOT / filename)
        torch_requirements = [
            requirement
            for requirement in requirements
            if requirement.lower().startswith("torch ")
        ]
        assert torch_requirements == [TORCH_CPU_PIN], filename


def test_docker_context_excludes_nested_python_runtime_artifacts() -> None:
    patterns = _active_requirements(ROOT / ".dockerignore")
    assert "**/__pycache__/" in patterns
    for extension in ("pyc", "pyo", "pyd"):
        assert f"**/*.{extension}" in patterns
