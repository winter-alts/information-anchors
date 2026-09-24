from __future__ import annotations

import numpy as np
import torch


class TorchKSG:
    """Exact batched KSG-1 with Chebyshev distances on a CUDA device.

    The future-side pairwise distance matrix is invariant under circular row
    shifts, so it is built once and indexed for every null pairing.
    """

    def __init__(
        self,
        future: np.ndarray,
        *,
        k: int,
        device: str,
        shift_batch_size: int = 8,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        future_array = np.asarray(future, dtype=np.float64)
        if future_array.ndim == 1:
            future_array = future_array[:, None]
        if future_array.ndim != 2 or not np.isfinite(future_array).all():
            raise ValueError("Future array must be finite [samples, features].")
        if not 1 <= k < len(future_array):
            raise ValueError("KSG k must satisfy 1 <= k < num_samples.")
        if shift_batch_size < 1:
            raise ValueError("shift_batch_size must be positive.")
        self.device = torch.device(device)
        self.dtype = dtype
        self.k = int(k)
        self.n_samples = int(len(future_array))
        self.shift_batch_size = int(shift_batch_size)
        self.future = torch.as_tensor(future_array, device=self.device, dtype=dtype)
        self.future_distances = self._chebyshev_distances(self.future)
        self.row_index = torch.arange(self.n_samples, device=self.device)
        self.constant = torch.special.digamma(
            torch.as_tensor(float(self.k), device=self.device, dtype=dtype)
        ) + torch.special.digamma(
            torch.as_tensor(float(self.n_samples), device=self.device, dtype=dtype)
        )

    @staticmethod
    def _chebyshev_distances(values: torch.Tensor) -> torch.Tensor:
        return (values[:, None, :] - values[None, :, :]).abs().amax(dim=-1)

    def _estimate_batch(
        self,
        x_distances: torch.Tensor,
        shifts: torch.Tensor,
    ) -> torch.Tensor:
        permutations = torch.remainder(
            self.row_index[None, :] - shifts[:, None], self.n_samples
        )
        return self._estimate_permutation_batch(x_distances, permutations)

    def _estimate_permutation_batch(
        self,
        x_distances: torch.Tensor,
        permutations: torch.Tensor,
    ) -> torch.Tensor:
        if permutations.ndim != 2 or permutations.shape[1] != self.n_samples:
            raise ValueError(
                "Permutations must have shape [batch, num_samples], got "
                f"{tuple(permutations.shape)}."
            )
        future_rows = self.future_distances[permutations]
        future_distances = torch.gather(
            future_rows,
            2,
            permutations[:, None, :].expand(-1, self.n_samples, -1),
        )
        joint_distances = torch.maximum(x_distances[None, :, :], future_distances)
        radii = torch.kthvalue(joint_distances, self.k + 1, dim=-1).values
        radii = torch.nextafter(radii, torch.zeros_like(radii))
        x_counts = (x_distances[None, :, :] <= radii[:, :, None]).sum(dim=-1) - 1
        y_counts = (future_distances <= radii[:, :, None]).sum(dim=-1) - 1
        estimates = self.constant - torch.mean(
            torch.special.digamma(x_counts.to(self.dtype) + 1.0) + torch.special.digamma(y_counts.to(self.dtype) + 1.0),
            dim=-1,
        )
        return estimates

    def estimate_observed_and_shifts(
        self,
        x: np.ndarray,
        shifts: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        x_array = np.asarray(x, dtype=np.float64)
        if x_array.ndim == 1:
            x_array = x_array[:, None]
        if x_array.shape[0] != self.n_samples or not np.isfinite(x_array).all():
            raise ValueError("X must be finite and have the prepared sample count.")
        x_tensor = torch.as_tensor(x_array, device=self.device, dtype=self.dtype)
        x_distances = self._chebyshev_distances(x_tensor)
        all_shifts = np.concatenate([np.zeros(1, dtype=np.int64), np.asarray(shifts, dtype=np.int64)])
        estimates = []
        with torch.no_grad():
            for start in range(0, len(all_shifts), self.shift_batch_size):
                batch = torch.as_tensor(
                    all_shifts[start : start + self.shift_batch_size],
                    device=self.device,
                    dtype=torch.long,
                )
                estimates.append(self._estimate_batch(x_distances, batch).cpu())
        values = torch.cat(estimates).numpy()
        return float(values[0]), values[1:].astype(np.float64, copy=False)

    def estimate_observed_and_permutations(
        self,
        x: np.ndarray,
        permutations: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Estimate observed MI and null MI for explicit target-row permutations."""
        x_array = np.asarray(x, dtype=np.float64)
        if x_array.ndim == 1:
            x_array = x_array[:, None]
        permutation_array = np.asarray(permutations, dtype=np.int64)
        if x_array.shape[0] != self.n_samples or not np.isfinite(x_array).all():
            raise ValueError("X must be finite and have the prepared sample count.")
        if permutation_array.ndim != 2 or permutation_array.shape[1] != self.n_samples:
            raise ValueError(
                "permutations must have shape [repetitions, num_samples]."
            )
        expected = np.arange(self.n_samples, dtype=np.int64)
        if any(not np.array_equal(np.sort(row), expected) for row in permutation_array):
            raise ValueError("Every explicit permutation must contain each row exactly once.")
        x_tensor = torch.as_tensor(x_array, device=self.device, dtype=self.dtype)
        x_distances = self._chebyshev_distances(x_tensor)
        all_permutations = np.concatenate(
            [expected[None, :], permutation_array], axis=0
        )
        estimates = []
        with torch.no_grad():
            for start in range(0, len(all_permutations), self.shift_batch_size):
                batch = torch.as_tensor(
                    all_permutations[start : start + self.shift_batch_size],
                    device=self.device,
                    dtype=torch.long,
                )
                estimates.append(
                    self._estimate_permutation_batch(x_distances, batch).cpu()
                )
        values = torch.cat(estimates).numpy()
        return float(values[0]), values[1:].astype(np.float64, copy=False)
