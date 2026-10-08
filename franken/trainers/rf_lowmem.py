import logging
import warnings
from pathlib import Path
from typing import Literal, Mapping

import torch
import torch.utils.data
from torch import Tensor

from franken.trainers.rf_trainer import RandomFeaturesTrainer
import franken.utils.distributed as dist_utils
from franken.data.base import Configuration, Target, TargetType, is_scalar_target
from franken.rf.model import FrankenPotential
from franken.utils.linalg.cov import (
    lowmem_normalize_leading_eig,
    rank1_update,
    rankk_update,
)
from franken.utils.linalg.tri import pack_upper, triangular_lerp, unpack_upper
from franken.utils.misc import no_jit, throughput

try:
    import cupy.cuda
    from cupy_backends.cuda.libs import cublas, cusolver
except ImportError:
    cupy = None
    cusolver = None
    cublas = None


logger = logging.getLogger("franken")


class LowMemRandomFeaturesTrainer(RandomFeaturesTrainer):
    """Low-memory variant of :class:`franken.trainers.RandomFeaturesTrainer` random-features trainer.

    To reduce the memory footprint, only 2 training targets are allowed (e.g. energy & forces or
    forces & stress, etc.).

    All other arguments and behavior is the same as :class:`franken.trainers.RandomFeaturesTrainer`.
    """

    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        l2_penalty: float | list[float],
        training_targets: list[TargetType],
        target_weight: Mapping[TargetType, float | list[float]],
        random_features_normalization: Literal["leading_eig"] | None = "leading_eig",
        log_dir: Path | None = None,
        save_every_model: bool = True,
        device: torch.device | str | int = "cuda:0",
        dtype: str | torch.dtype = torch.float32,
        save_fmaps: bool = True,
        metrics: list[str] | None = None,
        save_cho_factor: bool = True,
    ):
        if len(training_targets) != 2:
            raise ValueError(
                f"The low-memory trainer only works with 2 training targets. "
                f"Found {len(training_targets)} targets, please use "
                f"franken.trainers.RandomFeaturesTrainer instead."
            )
        super().__init__(
            train_dataloader=train_dataloader,
            l2_penalty=l2_penalty,
            training_targets=training_targets,
            target_weight=target_weight,
            random_features_normalization=random_features_normalization,
            log_dir=log_dir,
            save_every_model=save_every_model,
            device=device,
            dtype=dtype,
            save_fmaps=save_fmaps,
            metrics=metrics,
            save_cho_factor=save_cho_factor,
        )

    def psd_solve(
        self,
        cov: torch.Tensor,
        rhs: torch.Tensor,
        penalty: float,
        return_cho_factor: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""Solve ridge regression with the low-memory CUDA solver when available.

        Multiple right-hand sides are supported.
        Instead of providing the data matrix (commonly :math:`X` in ridge-regression notation),
        and labels (commonly :math:`y`), we are given directly :math:`\text{cov} = X^{\top} X`
        and :math:`\text{rhs} = X^{\top} y`.
        Since :attr:`cov` is symmetric only its **upper triangle** will be accessed.

        To limit memory usage, the :attr:`cov` matrix **may be overwritten**, and :math:`rhs`
        may also be overwritten (depending on its memory layout).

        Args:
            cov (Tensor): covariance of the linear system
            rhs (Tensor): right hand side (one or more) of the linear system
            penalty (float): Tikhonov l2 penalty
            return_cho_factor (bool): Also return the packed upper Cholesky factor.

        Returns:
            The ridge regression coefficients, or (coefficients, packed factor)
            when return_cho_factor is True.
        """
        if cupy is None or cov.device.type != "cuda":
            if cov.device.type == "cuda":
                warnings.warn(
                    "low-memory solver cannot be used because `cupy` is not available. "
                    "Install `cupy` if you encounter memory problems."
                )
            return super().psd_solve(cov, rhs, penalty, return_cho_factor)

        assert cusolver is not None and cublas is not None and cupy is not None
        assert cov.device.type == "cuda"
        dtype = cov.dtype
        n = cov.shape[0]

        # Add diagonal without copies
        cov.diagonal().add_(penalty)

        if dtype == torch.float32:
            potrf = cusolver.spotrf
            potrf_bufferSize = cusolver.spotrf_bufferSize
            potrs = cusolver.spotrs
        elif dtype == torch.float64:
            potrf = cusolver.dpotrf
            potrf_bufferSize = cusolver.dpotrf_bufferSize
            potrs = cusolver.dpotrs
        else:
            raise ValueError(dtype)

        # cov must be f-contiguous (column-contiguous, stride is (1, n))
        assert cov.dim() == 2
        assert cov.shape[0] == cov.shape[1]
        transpose = False
        if n != 1:
            if cov.stride(0) != 1:
                cov = cov.T
                transpose = True
        assert cov.stride(0) == 1
        cov_cp = cupy.asarray(cov)

        # save rhs shape to restore it later on.
        rhs_shape = rhs.shape
        rhs = rhs.reshape(n, -1)
        n_rhs = rhs.shape[1]
        if rhs.stride(0) != 1:  # force rhs to be f-contiguous
            # `contiguous` causes a copy
            rhs = rhs.T.contiguous().T
        assert rhs.stride(0) == 1
        rhs_cp = cupy.asarray(rhs)

        handle = cupy.cuda.device.get_cusolver_handle()
        uplo = (
            cublas.CUBLAS_FILL_MODE_LOWER
            if transpose
            else cublas.CUBLAS_FILL_MODE_UPPER
        )
        dev_info = torch.empty(
            1, dtype=torch.int32
        )  # don't allocate with cupy as it uses a separate mem pool
        dev_info_cp = cupy.asarray(dev_info)

        worksize = potrf_bufferSize(handle, uplo, n, cov_cp.data.ptr, n)
        workspace = torch.empty(worksize, dtype=dtype)
        workspace_cp = cupy.asarray(workspace)

        # Cholesky factorization
        potrf(
            handle,
            uplo,
            n,
            cov_cp.data.ptr,
            n,
            workspace_cp.data.ptr,
            worksize,
            dev_info_cp.data.ptr,
        )
        if (dev_info_cp != 0).any():
            raise torch.linalg.LinAlgError(
                f"Error reported by {potrf.__name__} in cuSOLVER. devInfo = {dev_info_cp}."
            )

        # Solve: A * X = B
        potrs(
            handle,
            uplo,
            n,
            n_rhs,
            cov_cp.data.ptr,
            n,
            rhs_cp.data.ptr,
            n,
            dev_info_cp.data.ptr,
        )
        if (dev_info_cp != 0).any():
            raise torch.linalg.LinAlgError(
                f"Error reported by {potrf.__name__} in cuSOLVER. devInfo = {dev_info_cp}."
            )

        solution = torch.as_tensor(rhs).reshape(rhs_shape)
        if return_cho_factor:
            # Restore the input orientation: its upper triangle now holds the factor.
            return solution, pack_upper(cov.T if transpose else cov).detach()
        return solution

    def _offload_covs_and_coeffs(self, covs, coeffs):
        """Cache both covariance triangles, their diagonals, and coefficients on CPU."""
        covariance = next(iter(covs.values()))[0].detach().to("cpu")
        packed_upper = pack_upper(covariance)
        packed_lower = pack_upper(covariance.T)
        self._covs_cache, self._coeffs_cache = (
            {
                target: (packed_upper, packed_lower, diagonal.detach().to("cpu"), upper)
                for target, (_, diagonal, upper) in covs.items()
            },
            {target: coeff.detach().to("cpu") for target, coeff in coeffs.items()},
        )

    def _restore_covs_and_coeffs(self, n_features: int):
        """Restore the shared covariance layout and coefficients on the trainer device."""
        if self._covs_cache is None or self._coeffs_cache is None:
            raise RuntimeError("Covariance and coefficient caches are not available")
        packed_upper, packed_lower, _, _ = next(iter(self._covs_cache.values()))
        covariance = unpack_upper(packed_upper, n_features).to(self.device)
        covariance.add_(unpack_upper(packed_lower, n_features).T.to(self.device))
        covariance.diagonal().zero_()
        return (
            {
                target: (covariance, diagonal.to(self.device), upper)
                for target, (_, _, diagonal, upper) in self._covs_cache.items()
            },
            {
                target: coeff.to(self.device)
                for target, coeff in self._coeffs_cache.items()
            },
        )

    @no_jit()
    @torch.no_grad()
    def _covs_and_coeffs(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        model: FrankenPotential,
        dataloader: torch.utils.data.DataLoader,
    ):
        tot_dset_size = len(dataloader.dataset)  # type: ignore
        n_rf = model.rf.total_random_features

        covariance = torch.zeros((n_rf, n_rf), device=self.device, dtype=self.buffer_dt)
        diags = [
            torch.zeros((n_rf,), device=self.device, dtype=self.buffer_dt)
            for _ in range(2)
        ]
        coeffs = [
            torch.zeros((n_rf,), device=self.device, dtype=self.buffer_dt)
            for _ in range(2)
        ]
        cov_upper = [True, False]

        # NOTE: the feature maps are never synced between devices.
        #       They can only used correctly by iterating through the
        #       same dataloader as here, using the same method. Otherwise
        #       they may not be in the correct order
        self.fmaps: dict[TargetType, list[Tensor]] = {
            t: [] for t in self.training_targets
        }
        progress_bar = throughput(
            dataloader,
            desc="covs+coeffs",
            total=tot_dset_size,
            device=self.device,
        )
        for i, (data, targets) in enumerate(progress_bar):
            assert isinstance(data, Configuration)
            data = data.to(device=self.device)
            assert data.natoms.numel() == 1, "Batched training is not supported"
            targets: Target = targets.to(device=self.device)

            target_fmaps = model.grad_feature_map(data, self.training_targets)
            if self.save_fmaps and i == 0:
                self.warn_save_fmaps(list(target_fmaps.values()), len(dataloader))

            for j, tgt_name in enumerate(self.training_targets):
                try:
                    tgt = targets[tgt_name]
                except KeyError:
                    raise RuntimeError(
                        f"Target {i} does not contain any values for {tgt_name}."
                    )
                tgt_per_atom = (tgt / data.natoms).to(dtype=self.buffer_dt)
                fmap = target_fmaps[tgt_name].to(self.buffer_dt)
                if is_scalar_target(tgt_name):
                    rank1_update(
                        covariance, diags[j], fmap.reshape(-1), upper=cov_upper[j]
                    )
                    coeffs[j].add_(fmap.reshape(-1), alpha=tgt_per_atom.item())
                else:
                    rankk_update(covariance, diags[j], fmap, upper=cov_upper[j])
                    coeffs[j].addmv_(fmap, tgt_per_atom.reshape(-1))
                if self.save_fmaps:
                    self.fmaps[tgt_name].append(fmap)
        # Sync covariance matrices & coefficients
        for tgt_name in self.training_targets:
            dist_utils.all_sum(covariance)
            for j in range(2):
                dist_utils.all_sum(diags[j])
                dist_utils.all_sum(coeffs[j])
        # RF normalization
        if self.random_features_normalization == "leading_eig":
            logger.warning(
                "`leading_eig` normalization has high memory usage. If you encounter OOM errors try to disable it."
            )
            for j in range(2):
                lowmem_normalize_leading_eig(
                    covariance, diags[j], coeffs[j], upper=cov_upper[j]
                )
        elif self.random_features_normalization is not None:
            raise NotImplementedError(
                f"Covariance normalization {self.random_features_normalization} is not implemented."
            )
        covs_out = {
            self.training_targets[0]: (covariance, diags[0], cov_upper[0]),
            self.training_targets[1]: (covariance, diags[1], cov_upper[1]),
        }
        coeffs_out = {
            self.training_targets[0]: coeffs[0],
            self.training_targets[1]: coeffs[1],
        }
        return covs_out, coeffs_out

    @torch.no_grad()
    def solve(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        covs: dict[TargetType, tuple[Tensor, Tensor, bool]],
        coeffs: dict[TargetType, Tensor],
        l2_penalty: float = 1e-6,
        return_cho_factor: bool = False,
        **weights,
    ) -> Tensor | tuple[Tensor, Tensor]:
        target_weight = {}
        for k, v in weights.items():
            target_weight[k.split("_")[0]] = v
        weights_norm_factor = sum(target_weight.values())

        diag_upper, diag_lower, cov, weight_lower = None, None, None, None
        coeff_upper, coeff_lower = None, None
        for tt in self.training_targets:
            if covs[tt][2]:
                diag_upper = covs[tt][1]
                cov = covs[tt][0]
                coeff_upper = coeffs[tt]
            else:
                diag_lower = covs[tt][1]
                coeff_lower = coeffs[tt]
                weight_lower = target_weight[tt] / weights_norm_factor
        assert (
            diag_upper is not None
            and diag_lower is not None
            and cov is not None
            and weight_lower is not None
            and coeff_upper is not None
            and coeff_lower is not None
        )

        # This is the 2nd copy of the covariance matrix that we need to store.
        lerped_cov, lerped_diag = triangular_lerp(
            cov,
            diag_upper=diag_upper,
            diag_lower=diag_lower,
            weight=weight_lower,
            inplace=False,
        )
        lerped_cov.diagonal().copy_(lerped_diag)
        rhs = torch.lerp(coeff_upper, coeff_lower, weight_lower)
        return self.psd_solve(lerped_cov, rhs, l2_penalty, return_cho_factor)
