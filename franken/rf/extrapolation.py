"""D-optimality extrapolation grades for random-feature potentials."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch

from franken.data import Configuration


@dataclass
class ActiveSet:
    """Species-specific active random-feature rows and inverse factors.

    The active matrix for each species has shape ``[n_active, n_features]``. Its
    inverse factor has shape ``[n_features, n_active]`` so that an atomic feature
    row ``phi`` can be scored as ``phi @ inverse`` for both square and
    underdetermined active sets.
    """

    active_matrices: dict[int, torch.Tensor]
    inverse_matrices: dict[int, torch.Tensor]
    regularization: float = 1e-8
    selection_method: str = "maxvol"

    @classmethod
    def from_matrices(
        cls,
        active_matrices: dict[int, torch.Tensor],
        regularization: float = 1e-8,
        selection_method: str = "maxvol",
    ) -> "ActiveSet":
        inverse_matrices = {
            species: ExtrapolationGrade.inverse_active_matrix(
                active_matrix, regularization=regularization
            )
            for species, active_matrix in active_matrices.items()
        }
        return cls(
            active_matrices=active_matrices,
            inverse_matrices=inverse_matrices,
            regularization=regularization,
            selection_method=selection_method,
        )

    @classmethod
    def load(
        cls,
        path: os.PathLike | str,
        map_location: torch.device | str | None = None,
    ) -> "ActiveSet":
        data = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(data, cls):
            return data
        return cls(
            active_matrices=data["active_matrices"],
            inverse_matrices=data["inverse_matrices"],
            regularization=data.get("regularization", 1e-8),
            selection_method=data.get("selection_method", "maxvol"),
        )

    def save(self, path: os.PathLike | str) -> None:
        torch.save(
            {
                "active_matrices": self.active_matrices,
                "inverse_matrices": self.inverse_matrices,
                "regularization": self.regularization,
                "selection_method": self.selection_method,
            },
            path,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ActiveSet":
        return ActiveSet(
            active_matrices={
                species: matrix.to(device=device, dtype=dtype)
                for species, matrix in self.active_matrices.items()
            },
            inverse_matrices={
                species: matrix.to(device=device, dtype=dtype)
                for species, matrix in self.inverse_matrices.items()
            },
            regularization=self.regularization,
            selection_method=self.selection_method,
        )

    @property
    def species(self) -> list[int]:
        return sorted(self.active_matrices)


class ExtrapolationGrade:
    """Build active sets and compute RF extrapolation grades."""

    def __init__(self, active_set: ActiveSet):
        self.active_set = active_set

    @staticmethod
    def _as_configuration(data: Configuration | tuple) -> Configuration:
        if isinstance(data, Configuration):
            return data
        if (
            isinstance(data, tuple)
            and len(data) > 0
            and isinstance(data[0], Configuration)
        ):
            return data[0]
        raise TypeError(
            f"Expected a Configuration or (Configuration, ...). Got {type(data)}."
        )

    @staticmethod
    def inverse_active_matrix(
        active_matrix: torch.Tensor,
        regularization: float = 1e-8,
    ) -> torch.Tensor:
        """Return a right inverse or regularized pseudo-inverse factor."""
        if active_matrix.ndim != 2:
            raise ValueError(
                f"Active matrix must be 2D. Got shape {active_matrix.shape}."
            )
        if active_matrix.shape[0] == 0:
            raise ValueError("Cannot invert an empty active matrix.")

        work_matrix = (
            active_matrix.to(dtype=torch.float64)
            if active_matrix.dtype in {torch.float16, torch.float32, torch.bfloat16}
            else active_matrix
        )

        n_active, n_features = work_matrix.shape
        if n_active == n_features and regularization <= 0:
            return torch.linalg.inv(work_matrix)
        if regularization <= 0:
            return torch.linalg.pinv(work_matrix)

        if n_active <= n_features:
            gram = work_matrix @ work_matrix.T
            eye = torch.eye(
                n_active, dtype=work_matrix.dtype, device=work_matrix.device
            )
            gram = gram + regularization * eye
            return work_matrix.T @ torch.linalg.pinv(gram)

        gram = work_matrix.T @ work_matrix
        eye = torch.eye(n_features, dtype=work_matrix.dtype, device=work_matrix.device)
        gram = gram + regularization * eye
        return torch.linalg.pinv(gram) @ work_matrix.T

    @staticmethod
    def _select_active_rows_pivoted_qr(
        rows: torch.Tensor,
        max_rows: int | None = None,
        tolerance: float = 1e-12,
    ) -> torch.Tensor:
        """Select linearly informative rows with a greedy pivoted-QR heuristic."""
        rows_work = (
            rows.to(dtype=torch.float64)
            if rows.dtype in {torch.float16, torch.float32, torch.bfloat16}
            else rows
        )
        n_rows, n_features = rows.shape
        if max_rows is None:
            max_rows = n_features
        max_rows = min(max_rows, n_rows)
        if max_rows <= 0:
            raise ValueError(f"max_rows must be positive. Got {max_rows}.")

        selected: list[int] = []
        basis = rows_work.new_empty((0, n_features))
        row_indices = torch.arange(n_rows, device=rows.device)
        available = torch.ones(n_rows, dtype=torch.bool, device=rows.device)

        for _ in range(max_rows):
            residuals = rows_work
            if basis.numel() > 0:
                residuals = rows_work - (rows_work @ basis.T) @ basis
            scores = torch.linalg.vector_norm(residuals, dim=1)
            scores = torch.where(available, scores, torch.full_like(scores, -1))
            pivot = int(torch.argmax(scores).item())
            pivot_score = scores[pivot]
            if pivot_score <= tolerance:
                break

            selected.append(pivot)
            available[pivot] = False
            next_basis = residuals[pivot] / pivot_score
            basis = torch.cat((basis, next_basis.unsqueeze(0)), dim=0)

        if not selected:
            pivot = int(torch.argmax(torch.linalg.vector_norm(rows_work, dim=1)).item())
            selected.append(pivot)
        return row_indices[selected]

    @staticmethod
    def _select_active_rows_maxvol(
        rows: torch.Tensor,
        maxvol_tolerance: float = 1.01,
        maxvol_iters: int = 300,
        init_tolerance: float = 1e-12,
    ) -> torch.Tensor:
        """Select a square active matrix with the MaxVol row-exchange algorithm."""
        rows_work = (
            rows.to(dtype=torch.float64)
            if rows.dtype in {torch.float16, torch.float32, torch.bfloat16}
            else rows
        )
        n_rows, n_features = rows.shape
        if n_rows < n_features:
            return ExtrapolationGrade._select_active_rows_pivoted_qr(
                rows,
                max_rows=n_rows,
                tolerance=init_tolerance,
            )

        selected = ExtrapolationGrade._select_active_rows_pivoted_qr(
            rows_work,
            max_rows=n_features,
            tolerance=init_tolerance,
        )
        if selected.numel() < n_features:
            return selected

        selected = selected.clone()
        for _ in range(maxvol_iters):
            active_matrix = rows_work[selected]
            active_inverse = torch.linalg.inv(active_matrix)
            coefficients = rows_work @ active_inverse
            abs_coefficients = torch.abs(coefficients)
            max_value, flat_idx = torch.max(abs_coefficients.reshape(-1), dim=0)
            if max_value <= maxvol_tolerance:
                break

            row_idx = int(flat_idx.item() // n_features)
            active_row_idx = int(flat_idx.item() % n_features)
            selected[active_row_idx] = row_idx

        return selected

    @staticmethod
    def select_active_rows(
        rows: torch.Tensor,
        max_rows: int | None = None,
        method: str = "pivoted-qr",
        tolerance: float = 1e-12,
        maxvol_tolerance: float = 1.01,
        maxvol_iters: int = 300,
    ) -> torch.Tensor:
        """Select active rows using either pivoted QR or MaxVol."""
        if rows.ndim != 2:
            raise ValueError(f"Expected a 2D row matrix. Got shape {rows.shape}.")
        if rows.shape[0] == 0:
            raise ValueError("Cannot select active rows from an empty matrix.")

        n_features = rows.shape[1]
        if method == "pivoted-qr":
            return ExtrapolationGrade._select_active_rows_pivoted_qr(
                rows,
                max_rows=max_rows,
                tolerance=tolerance,
            )
        if method == "maxvol":
            underdetermined = rows.shape[0] < n_features
            if (
                max_rows is not None
                and max_rows != n_features
                and not (underdetermined and max_rows == rows.shape[0])
            ):
                raise ValueError(
                    "MaxVol selection requires a square active set, so max_rows "
                    f"must be None or {n_features}. Got {max_rows}."
                )
            return ExtrapolationGrade._select_active_rows_maxvol(
                rows,
                maxvol_tolerance=maxvol_tolerance,
                maxvol_iters=maxvol_iters,
                init_tolerance=tolerance,
            )
        raise ValueError(f"Unknown active-row selection method: {method}.")

    @classmethod
    def build_active_set_from_features(
        cls,
        atomic_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        max_rows: int | None = None,
        regularization: float = 1e-8,
        selection_method: str = "maxvol",
        maxvol_tolerance: float = 1.01,
        maxvol_iters: int = 300,
    ) -> ActiveSet:
        """Build a species-specific active set from precomputed atomic RF rows."""
        if selection_method not in ("pivoted-qr", "maxvol"):
            raise ValueError(
                f"Unknown active-row selection method: {selection_method}."
            )
        if atomic_features.ndim != 2:
            raise ValueError(
                f"Expected atomic features with shape [atoms, features]. Got {atomic_features.shape}."
            )
        if (
            atomic_numbers.ndim != 1
            or atomic_numbers.shape[0] != atomic_features.shape[0]
        ):
            raise ValueError(
                "atomic_numbers must be a 1D tensor with one entry per feature row."
            )

        active_matrices: dict[int, torch.Tensor] = {}
        for atomic_number in torch.unique(atomic_numbers, sorted=True):
            mask = atomic_numbers == atomic_number
            species_rows = atomic_features[mask]
            n_active = min(
                species_rows.shape[0],
                max_rows if max_rows is not None else species_rows.shape[1],
            )
            if (
                selection_method == "maxvol"
                and species_rows.shape[0] >= species_rows.shape[1]
            ):
                selected = cls.select_active_rows(
                    species_rows,
                    max_rows=max_rows,
                    method=selection_method,
                    maxvol_tolerance=maxvol_tolerance,
                    maxvol_iters=maxvol_iters,
                )
            else:
                selected = cls.select_active_rows(
                    species_rows,
                    max_rows=n_active,
                    method="pivoted-qr",
                )
            active_matrices[int(atomic_number.item())] = (
                species_rows[selected].detach().clone()
            )

        return ActiveSet.from_matrices(
            active_matrices,
            regularization=regularization,
            selection_method=selection_method,
        )

    @classmethod
    @torch.no_grad
    def build_active_set(
        cls,
        model: torch.nn.Module,
        data: Configuration | Iterable[Configuration | tuple],
        max_rows: int | None = None,
        regularization: float = 1e-8,
        device: torch.device | str | None = None,
        selection_method: str = "maxvol",
        maxvol_tolerance: float = 1.01,
        maxvol_iters: int = 300,
    ) -> ActiveSet:
        """Build an active set from a trained model and configurations."""
        if selection_method not in ("pivoted-qr", "maxvol"):
            raise ValueError(
                f"Unknown active-row selection method: {selection_method}."
            )
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = None

        configs: Iterable[Configuration | tuple]
        if isinstance(data, Configuration):
            configs = (data,)
        else:
            configs = data

        rows_by_species: dict[int, list[torch.Tensor]] = {}
        for config in configs:
            config = cls._as_configuration(config)
            if device is not None:
                config = config.to(device)
            atomic_features = model.atomic_feature_map(config)
            for atomic_number in torch.unique(config.atomic_numbers, sorted=True):
                mask = config.atomic_numbers == atomic_number
                rows_by_species.setdefault(int(atomic_number.item()), []).append(
                    atomic_features[mask].detach().cpu()
                )

        if not rows_by_species:
            raise ValueError("Cannot build an active set from no configurations.")

        active_matrices: dict[int, torch.Tensor] = {}
        for species, row_blocks in rows_by_species.items():
            species_rows = torch.cat(row_blocks, dim=0)
            n_active = min(
                species_rows.shape[0],
                max_rows if max_rows is not None else species_rows.shape[1],
            )
            if (
                selection_method == "maxvol"
                and species_rows.shape[0] >= species_rows.shape[1]
            ):
                selected = cls.select_active_rows(
                    species_rows,
                    max_rows=max_rows,
                    method=selection_method,
                    maxvol_tolerance=maxvol_tolerance,
                    maxvol_iters=maxvol_iters,
                )
            else:
                selected = cls.select_active_rows(
                    species_rows,
                    max_rows=n_active,
                    method="pivoted-qr",
                )
            active_matrices[species] = species_rows[selected].clone()

        return ActiveSet.from_matrices(
            active_matrices,
            regularization=regularization,
            selection_method=selection_method,
        )

    @staticmethod
    def grade_atom(phi: torch.Tensor, active_inverse: torch.Tensor) -> torch.Tensor:
        """Compute ``max(abs(phi @ inv(A_s)))`` for one atomic feature row."""
        phi_work = phi.to(device=active_inverse.device, dtype=active_inverse.dtype)
        gamma_vector = phi_work @ active_inverse
        return torch.max(torch.abs(gamma_vector))

    def grade_configuration(
        self,
        atomic_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        return_atomic: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute per-atom and max configuration extrapolation grades."""
        sample_inverse = next(iter(self.active_set.inverse_matrices.values()))
        atomic_grades = torch.empty(
            atomic_features.shape[0],
            dtype=torch.promote_types(atomic_features.dtype, sample_inverse.dtype),
            device=atomic_features.device,
        )
        for idx, (phi, atomic_number) in enumerate(
            zip(atomic_features, atomic_numbers, strict=True)
        ):
            species = int(atomic_number.item())
            if species not in self.active_set.inverse_matrices:
                raise ValueError(f"Missing active set for atomic number {species}.")
            atomic_grades[idx] = self.grade_atom(
                phi, self.active_set.inverse_matrices[species]
            )
        config_grade = torch.max(atomic_grades)
        if return_atomic:
            return config_grade, atomic_grades
        return config_grade

    @torch.no_grad
    def grade_model_configuration(
        self,
        model: torch.nn.Module,
        data: Configuration,
        return_atomic: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute extrapolation grades by first evaluating model RF rows."""
        atomic_features = model.atomic_feature_map(data)
        return self.grade_configuration(
            atomic_features,
            data.atomic_numbers,
            return_atomic=return_atomic,
        )
