from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def create_run_dir(output_root: str, experiment_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    return run_dir


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def validate_mi_score_artifact(
    score_path: str | Path,
    *,
    seq_len: int,
    pred_len: int,
    patch_len: int | None = None,
) -> dict[str, Any]:
    """Reject MI scores computed for a different forecasting window."""

    score_path = Path(score_path)
    metadata_path = (
        score_path
        if score_path.suffix.lower() == ".json"
        else score_path.with_suffix(".json")
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"MI score metadata is required at {metadata_path}; recompute the scores "
            f"for seq_len={seq_len}, pred_len={pred_len}."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"MI score metadata must be a JSON object: {metadata_path}")

    actual_seq_len = metadata.get("seq_len")
    actual_pred_len = metadata.get("pred_len", metadata.get("horizon"))
    missing = [
        name
        for name, value in (
            ("seq_len", actual_seq_len),
            ("pred_len/horizon", actual_pred_len),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"MI score metadata {metadata_path} is missing {', '.join(missing)}; "
            "recompute the scores with the current generator."
        )

    expected = (int(seq_len), int(pred_len))
    actual = (int(actual_seq_len), int(actual_pred_len))
    if actual != expected:
        raise ValueError(
            f"MI score window mismatch: {metadata_path} was computed for "
            f"seq_len={actual[0]}, pred_len={actual[1]}, but this downstream run "
            f"requests seq_len={expected[0]}, pred_len={expected[1]}. Recompute or "
            "select the matching MI score artifact."
        )
    if patch_len is not None:
        actual_patch_len = metadata.get("patch_len")
        if actual_patch_len is None or int(actual_patch_len) != int(patch_len):
            raise ValueError(
                f"MI score patch mismatch: {metadata_path} has patch_len={actual_patch_len}, "
                f"expected {patch_len}."
            )
    return metadata


def _git(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def capture_run_context(run_dir: Path, command: str) -> None:
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    git_payload = {
        "commit": _git(["git", "rev-parse", "HEAD"]),
        "branch": _git(["git", "branch", "--show-current"]),
        "status": _git(["git", "status", "--short"]),
    }
    write_json(run_dir / "git.json", git_payload)

    versions: dict[str, str] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for module_name in [
        "torch",
        "timesfm",
        "transformers",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "datasets",
        "matplotlib",
    ]:
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:  # pragma: no cover - diagnostic only
            versions[module_name] = f"unavailable: {error}"
    write_json(run_dir / "environment.json", versions)
