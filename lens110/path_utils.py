"""Path helpers for the reorganized Lens110 retargeting workspace."""

from __future__ import annotations

from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    """Find the workspace root from a script path or an installed copy."""
    path = (start or Path(__file__)).resolve()
    for parent in (path, *path.parents):
        if (parent / "projects").is_dir() and (parent / "frameworks").is_dir():
            return parent
    # Fallback for a standalone checkout of this GMR directory.
    return path.parents[4]


def lens110_mjcf() -> Path:
    """Return the canonical Lens110 MJCF, with standalone fallbacks."""
    root = repository_root()
    candidates = (
        root
        / "frameworks"
        / "shared"
        / "lens110_isaaclab"
        / "lens110"
        / "legged_lab_lbot"
        / "source"
        / "legged_lab"
        / "legged_lab"
        / "data"
        / "Robots"
        / "model_humanoid_lens110"
        / "mjcf"
        / "lens110_21dof.xml",
        root / "tools" / "retargeting" / "robot_retargeter" / "asset" / "robot" / "lens110" / "lens110_21dof.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No canonical Lens110 MJCF was found in the reorganized workspace")


def project_training_dir() -> Path:
    """Return the full-body project's canonical generated-training directory."""
    return repository_root() / "projects" / "01_dance_whole_body" / "data" / "training"
