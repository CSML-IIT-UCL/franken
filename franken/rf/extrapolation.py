"""D-optimality extrapolation grades for random-feature potentials."""

from __future__ import annotations

import os
from dataclasses import dataclass
import logging
from typing import Iterable

import torch
from tqdm import tqdm

from franken.data import Configuration

logger = logging.getLogger("franken")


@dataclass
class ExtrapolationGrade:
    """Measure how new atomic environments compare with a reference active set.

    Use :meth:`build_active_set` with a trained model and reference structures,
    or :meth:`build_active_set_from_features` with precomputed atomic features.
    Then call :meth:`grade_model_configuration` to score new structures (or attach
    it to a ASE FrankenCalculator).

    By default, it uses the maxvol algorithm to select a square active matrix of
    reference atomic features.

    Each atomic grade is the largest absolute coefficient expressing its feature
    row in the active set. The per-structure extrapolation is computed as the largest atomic one.

    Attributes:
        active_matrices: Dictionary of selected feature rows, shaped ``[n_active, n_features]``
            for each atomic number if ``per_species=True``, or a single shared set with key ``0`` otherwise.
        inverse_matrices: Dictionary of inverse (or pseudoinverse) factors used for scoring,
            shaped ``[n_features, n_active]``.
        regularization: used when computing the inverse factor.
        selection_method: Requested selection method, ``"maxvol"`` or
            ``"pivoted-qr"``.
        per_species: Whether each chemical species has its own active set.
    """

    active_matrices: dict[int, torch.Tensor]
    inverse_matrices: dict[int, torch.Tensor]
    regularization: float = 1e-8
    selection_method: str = "maxvol"
    per_species: bool = False

    @classmethod
    def from_matrices(
        cls,
        active_matrices: dict[int, torch.Tensor],
        regularization: float = 1e-8,
        selection_method: str = "maxvol",
        per_species: bool = True,
    ) -> "ExtrapolationGrade":
        """Create a scorer from already selected atomic feature rows.

        Args:
            active_matrices: Matrices shaped ``[n_active, n_features]``, keyed by
                atomic number, or by ``0`` when ``per_species=False``.
            regularization: Nonnegative inverse stabilization strength; zero
                uses an inverse or pseudoinverse without damping. Defaults to
                ``1e-8``, unlike the builders, which default to zero.
            selection_method: Selection label to store with the scorer. This
                method does not perform row selection.
            per_species: Whether the matrices are grouped by chemical species.

        Returns:
            A scorer containing the supplied rows and computed inverse factors.
        """
        inverse_matrices = {
            species: cls.inverse_active_matrix(
                active_matrix, regularization=regularization
            )
            for species, active_matrix in active_matrices.items()
        }
        return cls(
            active_matrices=active_matrices,
            inverse_matrices=inverse_matrices,
            regularization=regularization,
            selection_method=selection_method,
            per_species=per_species,
        )

    @classmethod
    def load(
        cls,
        path: os.PathLike | str,
        map_location: torch.device | str | None = None,
    ) -> "ExtrapolationGrade":
        """Load an active set previously written with :meth:`save`.

        Args:
            path: Saved scorer file, separate from the model checkpoint.
            map_location: Device onto which the saved tensors are loaded.

        Returns:
            The restored scorer, including its selection settings.
        """
        data = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(data, cls):
            return data
        return cls(
            active_matrices=data["active_matrices"],
            inverse_matrices=data["inverse_matrices"],
            regularization=data.get("regularization", 1e-8),
            selection_method=data.get("selection_method", "maxvol"),
            per_species=data.get("per_species", True),
        )

    def save(self, path: os.PathLike | str) -> None:
        """Save active rows, inverse factors, and settings to ``path``.

        The trained model is not included; retain its checkpoint separately.
        """
        torch.save(
            {
                "active_matrices": self.active_matrices,
                "inverse_matrices": self.inverse_matrices,
                "regularization": self.regularization,
                "selection_method": self.selection_method,
                "per_species": self.per_species,
            },
            path,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ExtrapolationGrade":
        """Return a scorer with its tensors converted to ``device`` and ``dtype``."""
        return type(self)(
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
            per_species=self.per_species,
        )

    @property
    def species(self) -> list[int]:
        """Return sorted atomic-number keys, or ``[0]`` for a shared active set."""
        return sorted(self.active_matrices)

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
        """Compute the factor that maps feature rows to active-set coefficients.

        Args:
            active_matrix: Selected rows of shape ``[n_active, n_features]``.
            regularization: Nonnegative damping strength. Zero uses the exact
                inverse for a square matrix (which must be nonsingular), or a
                pseudoinverse otherwise. Positive values stabilize poorly
                constrained directions but also change extrapolation grades.

        Returns:
            A factor of shape ``[n_features, n_active]``. Multiply an atomic
            feature row by this factor to obtain its active-set coefficients.
        """
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
        qr_tolerance: float = 1e-12,
        show_progress: bool = False,
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

        with tqdm(
            total=max_rows,
            desc="Selecting active rows (QR)",
            disable=not show_progress,
        ) as progress:
            for _ in range(max_rows):
                residuals = rows_work
                if basis.numel() > 0:
                    residuals = rows_work - (rows_work @ basis.T) @ basis
                scores = torch.linalg.vector_norm(residuals, dim=1)
                scores = torch.where(available, scores, torch.full_like(scores, -1))
                pivot = int(torch.argmax(scores).item())
                pivot_score = scores[pivot]
                if pivot_score <= qr_tolerance:
                    break

                selected.append(pivot)
                progress.update(1)
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
        qr_tolerance: float = 1e-12,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Select a square active matrix with the MaxVol row-exchange algorithm."""
        rows_work = (
            rows.to(dtype=torch.float64)
            if rows.dtype in {torch.float16, torch.float32, torch.bfloat16}
            else rows
        )
        n_rows, n_features = rows.shape
        if n_rows < n_features:
            logger.debug(
                "MaxVol: using pivoted-QR fallback "
                f"(rows={n_rows}, features={n_features})."
            )
            return ExtrapolationGrade._select_active_rows_pivoted_qr(
                rows,
                show_progress=show_progress,
                max_rows=n_rows,
                qr_tolerance=qr_tolerance,
            )

        selected = ExtrapolationGrade._select_active_rows_pivoted_qr(
            rows_work,
            show_progress=show_progress,
            max_rows=n_features,
            qr_tolerance=qr_tolerance,
        )
        if selected.numel() < n_features:
            return selected

        selected = selected.clone()
        converged = False
        if maxvol_iters <= 0:
            raise ValueError("maxvol_iters must be positive")
        with tqdm(desc="MaxVol iterations:", disable=not show_progress) as progress:
            for iteration in range(maxvol_iters):
                active_matrix = rows_work[selected]
                active_inverse = torch.linalg.inv(active_matrix)
                coefficients = rows_work @ active_inverse
                abs_coefficients = torch.abs(coefficients)
                max_value, flat_idx = torch.max(abs_coefficients.reshape(-1), dim=0)
                progress.update(1)
                if max_value <= maxvol_tolerance:
                    converged = True
                    break

                row_idx = int(flat_idx.item() // n_features)
                active_row_idx = int(flat_idx.item() % n_features)
                selected[active_row_idx] = row_idx

        if converged:
            logger.debug("MaxVol converged after %d exchanges", iteration)
        else:
            logger.warning(
                "MaxVol did not converge after %d iterations (maximum coefficient %g, tolerance %g)",
                maxvol_iters,
                max_value,
                maxvol_tolerance,
            )

        return selected

    @staticmethod
    def select_active_rows(
        rows: torch.Tensor,
        max_rows: int | None = None,
        method: str = "pivoted-qr",
        qr_tolerance: float = 1e-12,
        maxvol_tolerance: float = 1.01,
        maxvol_iters: int = 300,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Select representative rows from a matrix of candidate atomic features.

        Args:
            rows: Candidate features of shape ``[n_atoms, n_features]``.
            max_rows: Maximum number of rows for QR; defaults to the feature
                dimension. MaxVol requires a square set, so this must
                be omitted or equal the feature dimension.
            method: ``"pivoted-qr"`` greedily adds the row with the largest
                component outside the span already selected. ``"maxvol"`` starts
                with QR and exchanges rows to increase the absolute determinant.
                If too few independent rows are available, it returns the QR set.
            qr_tolerance: Absolute residual-norm threshold for stopping QR selection.
            maxvol_tolerance: Stop MaxVol when the largest absolute coefficient
                of the candidate rows in the active set is at most this value.
                Values just above one give tighter refinement; default ``1.01``.
            maxvol_iters: Maximum number of MaxVol iterations.
            show_progress: Show QR row counts and a MaxVol iteration counter.

        Returns:
            A one-dimensional tensor of selected indices into ``rows``.
        """
        if rows.ndim != 2:
            raise ValueError(f"Expected a 2D row matrix. Got shape {rows.shape}.")
        if rows.shape[0] == 0:
            raise ValueError("Cannot select active rows from an empty matrix.")

        n_features = rows.shape[1]
        if method == "pivoted-qr":
            return ExtrapolationGrade._select_active_rows_pivoted_qr(
                rows,
                show_progress=show_progress,
                max_rows=max_rows,
                qr_tolerance=qr_tolerance,
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
                show_progress=show_progress,
                maxvol_tolerance=maxvol_tolerance,
                maxvol_iters=maxvol_iters,
                qr_tolerance=qr_tolerance,
            )
        raise ValueError(f"Unknown active-row selection method: {method}.")

    @classmethod
    def build_active_set_from_features(
        cls,
        atomic_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        max_rows: int | None = None,
        regularization: float = 0,
        selection_method: str = "maxvol",
        maxvol_tolerance: float = 1.01,
        maxvol_iters: int = 300,
        per_species: bool = True,
        show_progress: bool = True,
    ) -> ExtrapolationGrade:
        """Build a scorer from reference atomic features.

        Args:
            atomic_features: Unpooled RF rows of shape ``[n_atoms, n_features]``,
                as returned by the model's ``atomic_feature_map`` method.
            atomic_numbers: Atomic number for each row, shaped ``[n_atoms]``.
            max_rows: Maximum rows per active set; defaults to the feature
                dimension. A smaller limit uses QR even when MaxVol is requested.
            regularization: Nonnegative inverse stabilization strength; default
                zero. Positive values damp poorly constrained directions and
                change grades, but do not affect row selection.
            selection_method: ``"maxvol"`` refines a QR initialization through
                volume-increasing row exchanges. ``"pivoted-qr"`` uses only QR.
                Too few independent rows cause MaxVol to retain the QR set.
            maxvol_tolerance: MaxVol coefficient threshold; default ``1.01``.
                Values closer to one require tighter convergence.
            maxvol_iters: Maximum number of MaxVol iterations per active set.
            per_species: Build a separate active set for each chemical species
                (default), or one shared set when false.
            show_progress: Display simple progress bars; false disables them.

        Returns:
            A scorer containing selected reference rows and inverse factors.
            Tensors remain on the input feature device.
        """
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
        if not per_species:
            # Key 0 stores the global active set.
            n_active = min(
                atomic_features.shape[0],
                max_rows if max_rows is not None else atomic_features.shape[1],
            )
            if (
                selection_method == "maxvol"
                and atomic_features.shape[0] >= atomic_features.shape[1]
                and (max_rows is None or max_rows == atomic_features.shape[1])
            ):
                selected = cls.select_active_rows(
                    atomic_features,
                    show_progress=show_progress,
                    max_rows=max_rows,
                    method=selection_method,
                    maxvol_tolerance=maxvol_tolerance,
                    maxvol_iters=maxvol_iters,
                )
            else:
                selected = cls.select_active_rows(
                    atomic_features,
                    show_progress=show_progress,
                    max_rows=n_active,
                    method="pivoted-qr",
                )
            active_matrices[0] = atomic_features[selected].detach().clone()
        else:
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
                    and (max_rows is None or max_rows == species_rows.shape[1])
                ):
                    selected = cls.select_active_rows(
                        species_rows,
                        show_progress=show_progress,
                        max_rows=max_rows,
                        method=selection_method,
                        maxvol_tolerance=maxvol_tolerance,
                        maxvol_iters=maxvol_iters,
                    )
                else:
                    selected = cls.select_active_rows(
                        species_rows,
                        show_progress=show_progress,
                        max_rows=n_active,
                        method="pivoted-qr",
                    )
                active_matrices[int(atomic_number.item())] = (
                    species_rows[selected].detach().clone()
                )

        return cls.from_matrices(
            active_matrices,
            regularization=regularization,
            selection_method=selection_method,
            per_species=per_species,
        )

    @classmethod
    @torch.no_grad
    def build_active_set(
        cls,
        model: torch.nn.Module,
        data: Configuration | Iterable[Configuration | tuple],
        max_rows: int | None = None,
        regularization: float = 0,
        device: torch.device | str | None = None,
        selection_method: str = "maxvol",
        maxvol_tolerance: float = 1.01,
        maxvol_iters: int = 300,
        per_species: bool = True,
        show_progress: bool = True,
    ) -> ExtrapolationGrade:
        """Build a scorer from a trained model and reference configurations.

        Only structures are needed; energy and force labels are ignored. Atomic
        features are computed without gradients, collected on CPU, and selected
        separately for each species unless ``per_species=False``.

        Args:
            model: Trained model exposing ``atomic_feature_map(configuration)``.
            data: A single :class:`Configuration`, an iterable of configurations,
                or a dataset/dataloader yielding ``(Configuration, targets)``.
                ASE ``Atoms`` objects must first be converted to configurations.
                Dataset iteration may load targets; for unlabelled structures,
                yield ``dataset.__getitem__(i, no_targets=True)`` instead.
            max_rows: Maximum rows per active set; defaults to the feature
                dimension. A smaller limit uses QR even when MaxVol is requested.
            regularization: Nonnegative inverse stabilization strength; default
                zero. Positive values damp poorly constrained directions and
                change grades, but do not affect row selection.
            selection_method: ``"maxvol"`` refines a QR initialization through
                volume-increasing row exchanges. ``"pivoted-qr"`` uses only QR.
                Too few independent rows cause MaxVol to retain the QR set.
            maxvol_tolerance: MaxVol coefficient threshold; default ``1.01``.
                Values closer to one require tighter convergence.
            maxvol_iters: Maximum number of MaxVol iterations per active set.
            per_species: Build a separate active set for each chemical species
                (default), or one shared set when false.
            show_progress: Display simple progress bars; false disables them.
            device: Device for feature extraction, defaulting to the model's
                device. Move the model to that device before calling this method.

        Returns:
            A scorer with active rows and inverse factors stored on CPU. Use
            :meth:`to` to move it to another device for scoring.

        Example::

            scorer = ExtrapolationGrade.build_active_set(model, train_loader)
            grade = scorer.grade_model_configuration(model, configuration)
            scorer.save("extrapolation.pt")
        """
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
        all_rows: list[torch.Tensor] = []
        with tqdm(
            configs,
            desc="Computing atomic features",
            disable=not show_progress,
        ) as progress:
            for config in progress:
                config = cls._as_configuration(config)
                if device is not None:
                    config = config.to(device)
                atomic_features = model.atomic_feature_map(config)
                if per_species:
                    for atomic_number in torch.unique(
                        config.atomic_numbers, sorted=True
                    ):
                        mask = config.atomic_numbers == atomic_number
                        rows_by_species.setdefault(
                            int(atomic_number.item()), []
                        ).append(atomic_features[mask].detach().cpu())
                else:
                    all_rows.append(atomic_features.detach().cpu())

        if per_species and not rows_by_species:
            raise ValueError("Cannot build an active set from no configurations.")
        if not per_species and not all_rows:
            raise ValueError("Cannot build an active set from no configurations.")

        active_matrices: dict[int, torch.Tensor] = {}
        if not per_species:
            # Atomic numbers are positive, so key 0 stores the global active set.
            rows = torch.cat(all_rows, dim=0)
            n_active = min(
                rows.shape[0],
                max_rows if max_rows is not None else rows.shape[1],
            )
            if (
                selection_method == "maxvol"
                and rows.shape[0] >= rows.shape[1]
                and (max_rows is None or max_rows == rows.shape[1])
            ):
                selected = cls.select_active_rows(
                    rows,
                    show_progress=show_progress,
                    max_rows=max_rows,
                    method=selection_method,
                    maxvol_tolerance=maxvol_tolerance,
                    maxvol_iters=maxvol_iters,
                )
            else:
                selected = cls.select_active_rows(
                    rows,
                    show_progress=show_progress,
                    max_rows=n_active,
                    method="pivoted-qr",
                )
            active_matrices[0] = rows[selected].clone()
        else:
            for species, row_blocks in rows_by_species.items():
                species_rows = torch.cat(row_blocks, dim=0)
                n_active = min(
                    species_rows.shape[0],
                    max_rows if max_rows is not None else species_rows.shape[1],
                )
                if (
                    selection_method == "maxvol"
                    and species_rows.shape[0] >= species_rows.shape[1]
                    and (max_rows is None or max_rows == species_rows.shape[1])
                ):
                    selected = cls.select_active_rows(
                        species_rows,
                        show_progress=show_progress,
                        max_rows=max_rows,
                        method=selection_method,
                        maxvol_tolerance=maxvol_tolerance,
                        maxvol_iters=maxvol_iters,
                    )
                else:
                    selected = cls.select_active_rows(
                        species_rows,
                        show_progress=show_progress,
                        max_rows=n_active,
                        method="pivoted-qr",
                    )
                active_matrices[species] = species_rows[selected].clone()

        return cls.from_matrices(
            active_matrices,
            regularization=regularization,
            selection_method=selection_method,
            per_species=per_species,
        )

    @staticmethod
    def grade_atom(phi: torch.Tensor, active_inverse: torch.Tensor) -> torch.Tensor:
        """Return the extrapolation grade of a single atomic feature.

        Args:
            phi: Unpooled atomic RF vector of shape ``[n_features]``.
            active_inverse: Matching inverse factor of shape
                ``[n_features, n_active]``.

        Returns:
            Scalar tensor ``max(abs(phi @ active_inverse))`` on the factor's
            device. Larger values indicate stronger extrapolation.
        """
        phi_work = phi.to(device=active_inverse.device, dtype=active_inverse.dtype)
        gamma_vector = phi_work @ active_inverse
        return torch.max(torch.abs(gamma_vector))

    def grade_configuration(
        self,
        atomic_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        return_atomic: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Score a structure using its precomputed atomic feature rows.

        Args:
            atomic_features: Unpooled rows of shape ``[n_atoms, n_features]``
                computed with the same model used to build the active set.
            atomic_numbers: Atomic number for each feature row. Every species
                must have an active set when ``per_species=True``.
            return_atomic: Also return individual grades in input atom order.

        Returns:
            A scalar tensor containing the largest atomic grade. When
            ``return_atomic=True``, returns ``(grade, atomic_grades)`` with
            ``atomic_grades`` shaped ``[n_atoms]``. Results are on the feature
            tensor's device. Grades measure extrapolation, not prediction errors.
        """
        sample_inverse = next(iter(self.inverse_matrices.values()))
        atomic_grades = torch.empty(
            atomic_features.shape[0],
            dtype=torch.promote_types(atomic_features.dtype, sample_inverse.dtype),
            device=atomic_features.device,
        )
        if not self.per_species:
            if 0 not in self.inverse_matrices:
                raise ValueError("Missing global active set at key 0.")
            active_inverse = self.inverse_matrices[0]
            for idx, phi in enumerate(atomic_features):
                atomic_grades[idx] = self.grade_atom(phi, active_inverse)
            config_grade = torch.max(atomic_grades)
            if return_atomic:
                return config_grade, atomic_grades
            return config_grade

        for idx, (phi, atomic_number) in enumerate(
            zip(atomic_features, atomic_numbers, strict=True)
        ):
            species = int(atomic_number.item())
            if species not in self.inverse_matrices:
                raise ValueError(f"Missing active set for atomic number {species}.")
            atomic_grades[idx] = self.grade_atom(phi, self.inverse_matrices[species])
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
        """Score a new configuration with a trained model (without gradients).

        Args:
            model: Model with the same backbone, scaler, and RF parameters used
                to build this active set.
            data: Configuration on the model's device. Convert ASE structures
                using a dataset configured for this model's backbone.
            return_atomic: Also return grades for individual atomic environments.

        Returns:
            The largest atomic grade as a scalar tensor, or
            ``(grade, atomic_grades)`` when ``return_atomic=True``. See
            :meth:`grade_configuration` for interpretation and output shapes.
        """
        atomic_features = model.atomic_feature_map(data)
        return self.grade_configuration(
            atomic_features,
            data.atomic_numbers,
            return_atomic=return_atomic,
        )
