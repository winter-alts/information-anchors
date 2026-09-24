"""Local GPU allocation policy for expensive external-model runners."""

from __future__ import annotations


def validate_gpu_id(gpu_id: int) -> int:
    """Restrict official TS-RAG runs to physical GPUs 1 through 7."""
    value = int(gpu_id)
    if value < 1 or value > 7:
        raise ValueError(f"GPU id must be in the allowed physical range 1-7, got {value}")
    return value


def validate_device(device: str | None) -> str | None:
    """Reject CUDA aliases that could resolve to a forbidden physical GPU."""
    if device is None:
        return None
    value = str(device)
    if value == "cuda":
        raise ValueError("CUDA device must specify an allowed physical GPU id in 1-7")
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1]
        try:
            validate_gpu_id(int(suffix))
        except ValueError as exc:
            raise ValueError(
                f"device must use a CUDA GPU in the allowed physical range 1-7, got {value}"
            ) from exc
    return value
