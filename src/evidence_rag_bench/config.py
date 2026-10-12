"""Project-local paths used by the reproducible pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Resolved repository paths without creating files as a side effect."""

    project_root: Path
    artifacts_dir: Path
    corpus_dir: Path
    eval_dir: Path


def get_settings(project_root: Path | None = None, artifacts_dir: Path | None = None) -> Settings:
    """Return repository paths rooted at ``project_root`` or the current directory."""

    root = (project_root or Path.cwd()).resolve()
    return Settings(
        project_root=root,
        artifacts_dir=artifacts_dir.resolve() if artifacts_dir else root / "artifacts",
        corpus_dir=root / "data" / "corpus",
        eval_dir=root / "data" / "eval",
    )
