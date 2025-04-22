"""Random feature implementations for different kernels"""

import dataclasses
from math import ceil, sqrt

import numpy as np
import scipy as sp
import torch

from franken.config import GaussianRFConfig, MultiscaleGaussianRFConfig, RFConfig
from franken.utils.misc import sanitize_init_dict


class RandomFeaturesHead(torch.nn.Module):
    """Base class for random-feature heads"""

    def __init__(
        self,
        input_dim: int,
        num_random_features: int = 2**10,
        num_species: int | None = None,
        chemically_informed_ratio: float | None = None,
    ):
        r"""Initializer for random features base class.

        Args:
            input_dim (int): Dimension of the input features.
            num_random_features (int, optional): The number of random features to use in the feature mapping. Defaults to :math:`2^{10} = 1024`.
            num_species (int | None): The number of chemical species for
                which the kernel is computed. This parameter is relevant for systems
                with multiple chemical species. Defaults to :code:`None`.
            chemically_informed_ratio (float | None): The relative weight of chemically-informed kernels with respect to the all-species kernel. Ignored if
                :code:`num_species` is None. Defaults to :code:`None`.
        """
        super(RandomFeaturesHead, self).__init__()
        self.input_dim = input_dim
        self.num_random_features = num_random_features
        self.num_species = num_species
        self.chemically_informed_ratio = chemically_informed_ratio

        self._per_species_multiplier = 0
        if self.num_species is not None:
            assert self.num_species > 0
            self._per_species_multiplier += self.num_species
            if self.chemically_informed_ratio is not None:
                self._per_species_multiplier += 1
        else:
            self._per_species_multiplier += 1
        self.total_random_features = (
            self._per_species_multiplier * self.num_random_features
        )

        # Register weights buffer
        self.weights: torch.Tensor
        self.register_buffer("weights", torch.zeros((1, self.total_random_features)))

    def species_scatter_sum(
        self, Z: torch.Tensor, atomic_numbers: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Average features across all atoms in a configuration.

        Depending on the configuration of random features, this function will either perform
        a simple average, or will use a chemically-informed averaging method where features
        are averaged within each atomic type and concatenated across atomic types. In this latter
        case, the number of output features is larger than the number of input features. It will
        always be equal to :code:`self.total_random_features`.

        Args:
            Z (torch.Tensor): [num_atoms, feature_dim] tensor containing the random features for each atom
            atomic_numbers (torch.Tensor | None, optional): if specified, an integer tensor of size [num_atoms]
                detailing the atomic number of each atom in the configuration. Defaults to None.

        Returns:
            torch.Tensor: [total_feature_dim] tensor containing the random features for the whole configuration.
        """
        if self.num_species is None:
            return Z.mean(0)
        else:
            assert (
                atomic_numbers is not None
            ), "atomic_number should be specified when self.num_species is not None"
            species_map, scatter_idxs = torch.unique(
                atomic_numbers, sorted=True, return_inverse=True
            )
            assert (
                len(species_map) == self.num_species
            ), f"The provided atomic numbers {species_map}, are of a different number than self.num_species {self.num_species}"

            scatter_idxs = scatter_idxs.unsqueeze(-1).expand(
                Z.size()
            )  # ~[natoms, random_features_per_species] Broadcasting for backprop.
            chemically_informed_descriptors = (
                torch.zeros((self.num_species, self.num_random_features), dtype=Z.dtype)
                .to(Z.device)
                .scatter_reduce_(0, scatter_idxs, Z, "mean")
            ) / self.num_species  # Normalize.

            if self.chemically_informed_ratio is not None:
                kappa = self.chemically_informed_ratio
                assert (
                    0 <= kappa <= 1
                ), "The ratio of chemically informed feature feature map should be bounded between 0 and 1"
                chemically_informed_descriptors = torch.cat(
                    (
                        (1 - kappa) * Z.mean(0, keepdim=True),
                        kappa * chemically_informed_descriptors,
                    )
                )
            return chemically_informed_descriptors.view(-1)

    def init_args(self):
        """Returns the arguments needed to re-initialize this class."""
        return {
            "num_random_features": self.num_random_features,
            "num_species": self.num_species,
            "chemically_informed_ratio": self.chemically_informed_ratio,
        }


class OrthogonalRFF(RandomFeaturesHead):
    r"""
    `Orthogonal Random Fourier Features <https://arxiv.org/abs/1610.09072>`_ by Yu et al. for approximating the Gaussian kernel

    .. math::

        \text{exp}\left(-\frac{\| x - y \|^{2}}{2\ell^{2}}\right)

    .
    """

    def __init__(
        self,
        input_dim: int,
        num_random_features: int = 2**10,
        num_species: int | None = None,
        chemically_informed_ratio: float | None = None,
        use_offset: bool = True,
        length_scale: float = 1.0,
        rng_seed=None,
    ):
        r"""Initialize the ORFF approximation to the Gaussian kernel with given length scale :math:`\ell`.

        As with the Gaussian kernel, the RF approximation depends on the :code:`length_scale` parameter.
        In addition to that, one needs to choose the number of random features to control the approximation
        quality.

        Args:
            input_dim (int): Dimensionality of the input features.
            num_random_features (int, optional): The number of random features to use in the feature mapping. Defaults to :math:`2^{10} = 1024`.
            num_species (int | None): The number of chemical species for
                which the kernel is computed. This parameter is relevant for systems
                with multiple chemical species. Defaults to :code:`None`.
            chemically_informed_ratio (float | None): The relative weight of chemically-informed kernels with respect to the all-species kernel. Ignored if
                :code:`num_species` is None. Defaults to :code:`None`.
            use_offset (bool): A flag indicating whether to use an offset in
                the random feature generation. Using an offset reduces the number of
                random features by half but increases variance. Defaults to :code:`True`.
            length_scale (float): The length scale parameter :math:`\ell` that
                controls the smoothness of the kernel function. It affects how quickly
                the kernel values decay with distance. Defaults to 1.0.
            rng_seed (int | None): A seed for the random number generator
                used in generating random features. Setting this ensures reproducibility
                of results. Defaults to :code:`None`.

        """
        check_positive_arg(length_scale, "length_scale")
        super(OrthogonalRFF, self).__init__(
            input_dim,
            num_random_features,
            num_species=num_species,
            chemically_informed_ratio=chemically_informed_ratio,
        )
        self.rng_seed = rng_seed
        random_generator = get_seeded_generator(self.rng_seed)

        W = self._sample_orf_matrix(
            input_dim, self.num_random_features, random_generator, self.rng_seed
        )

        self.register_buffer("rff_matrix", W)
        self.register_buffer(
            "random_offset",
            torch.rand(self.num_random_features, generator=random_generator)
            * 2
            * np.pi,
        )
        self.register_buffer("length_scale", torch.tensor(length_scale))

        self.use_offset = use_offset
        if not self.use_offset:
            self.total_random_features = self.total_random_features * 2

    @staticmethod
    def _sample_orf_matrix(input_dim, num_random_features, torch_generator, rng_seed):
        num_folds = int(np.ceil(num_random_features / input_dim))

        G = torch.randn(
            num_folds,
            input_dim,
            input_dim,
            generator=torch_generator,
        )
        Q, _ = torch.linalg.qr(
            G, mode="complete"
        )  # The _columns_ in each batch of matrices in Q are orthonormal.

        Q = Q.transpose(
            -1, -2
        )  # The _rows_ in each batch of matrices in Q are orthonormal.

        S = torch.tensor(
            sp.stats.chi.rvs(
                input_dim,
                size=(num_folds, input_dim, 1),
                random_state=rng_seed,
            ),
        )

        W = Q * S  # [num_folds, input_dim, input_dim]
        W = torch.cat(
            [W[fold_idx, ...] for fold_idx in range(num_folds)], dim=0
        )  # Concatenate columns [num_heads, num_folds*input_dim, input_dim]
        W = W[:num_random_features, :]
        return W

    def feature_map(
        self,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the random-feature map for a given configuration :code:`h`

        Args:
            h (torch.Tensor):
                descriptors for a single configuration ~[natoms, descriptors]
            atomic_numbers (torch.Tensor):
                atomic numbers for a single configuration ~[natoms]
        """
        length_scale = self.length_scale
        # Convert dtype of rf matrices to same as h
        self.rff_matrix = self.rff_matrix.to(dtype=h.dtype)
        if self.use_offset:
            self.random_offset = self.random_offset.to(dtype=h.dtype)
        # Compute the orthogonal random features nonlinear mapping
        Z = torch.einsum("af, Ff -> aF", h, self.rff_matrix / length_scale)
        if self.use_offset:
            b = self.random_offset.unsqueeze(0)
            Z = torch.cos(Z + b) / sqrt(0.5 * self.num_random_features)
        else:
            _s = torch.sin(Z)
            _c = torch.cos(Z)
            Z = torch.cat((_s, _c), dim=-1) / sqrt(self.num_random_features)

        return self.species_scatter_sum(
            Z, atomic_numbers=atomic_numbers
        )  # Sum over the single species

    def init_args(self):
        return super().init_args() | {
            "use_offset": self.use_offset,
            "length_scale": self.length_scale.tolist(),
            "rng_seed": self.rng_seed,
        }


class BiasedOrthogonalRFF(OrthogonalRFF):
    def __init__(
        self,
        input_dim: int,
        num_random_features: int = 2**10,
        num_species: int | None = None,
        chemically_informed_ratio: float | None = None,
        use_offset: bool = True,
        length_scale: float = 1.0,
        bias: float = 1.0,
        rng_seed=None,
    ):
        super(BiasedOrthogonalRFF, self).__init__(
            input_dim,
            num_random_features,
            num_species=num_species,
            chemically_informed_ratio=chemically_informed_ratio,
            use_offset=use_offset,
            length_scale=length_scale,
            rng_seed=rng_seed,
        )
        self.register_buffer("bias", torch.tensor(bias))

    def feature_map(
        self,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.bias = self.bias.to(dtype=h.dtype)
        # unbiased_features ~ [num_random_features - 1]
        features = super().feature_map(h, atomic_numbers=atomic_numbers)
        assert features.ndim == 1
        features[-1] = torch.sqrt(self.bias)
        return features

    def init_args(self):
        return super().init_args() | {"bias": self.bias.tolist()}


class MultiScaleOrthogonalRFF(RandomFeaturesHead):
    r"""
    A multi-scale version of :class:`OrthogonalRFF` which splits the available random features among multiple length-scales.
    This approximates a mixture of Gaussian kernels at different scales, simplifying hyper-parameter tuning.

    .. math::

        \text{exp}\left(-\frac{\| x - y \|^{2}}{2\ell^{2}}\right)

    .
    """

    def __init__(
        self,
        input_dim: int,
        num_random_features: int = 2**10,
        num_species: int | None = None,
        chemically_informed_ratio: float | None = None,
        use_offset: bool = True,
        length_scale_low: float = 1.0,
        length_scale_high: float = 10.0,
        length_scale_num: int = 4,
        rng_seed=None,
    ):
        r"""Initialize the multi-scale ORFF approximation.

        Multiple scales are specified with arguments :code:`length_scale_low`, :code:`length_scale_high`
        and :code:`length_scale_num` which will be used to subdivide the available :code:`num_random_features`
        random features into :code:`length_scale_num` blocks with equally spaced length-scales.
        This means that each length-scale will only use a fraction of the total random features, but in practice
        we found this have very small impact on overall accuracy.
        This kernel can be seen as implicitly doing a grid-search over linearly spaced length-scales.

        Args:
            input_dim (int): Dimensionality of the input features.
            num_random_features (int, optional): The number of random features to use in the feature mapping. Defaults to :math:`2^{10} = 1024`.
            num_species (int | None): The number of chemical species for
                which the kernel is computed. This parameter is relevant for systems
                with multiple chemical species. Defaults to :code:`None`.
            chemically_informed_ratio (float | None): The relative weight of chemically-informed kernels with respect to the all-species kernel. Ignored if
                :code:`num_species` is None. Defaults to :code:`None`.
            use_offset (bool): A flag indicating whether to use an offset in
                the random feature generation. Using an offset reduces the number of
                random features by half but increases variance. Defaults to :code:`True`.
            length_scale_low (float): The lower end of the interval of length-scales considered.
            length_scale_high (float): The higher end of the interval of length-scales considered.
            length_scale_num (int): The number of different length-scales, equally spaced between
                :code:`length_scale_low` and :code:`length_scale_high` which are considered in the
                multi-scale approximation.
            rng_seed (int | None): A seed for the random number generator
                used in generating random features. Setting this ensures reproducibility
                of results. Defaults to :code:`None`.

        """
        super(MultiScaleOrthogonalRFF, self).__init__(
            input_dim,
            num_random_features,
            num_species=num_species,
            chemically_informed_ratio=chemically_informed_ratio,
        )
        self.rng_seed = rng_seed
        random_generator = get_seeded_generator(self.rng_seed)
        check_positive_arg(length_scale_low, "length_scale_low")
        self.length_scale_low = length_scale_low
        check_positive_arg(length_scale_high, "length_scale_high")
        self.length_scale_high = length_scale_high
        self.length_scale_num = length_scale_num

        W = OrthogonalRFF._sample_orf_matrix(
            input_dim, self.num_random_features, random_generator, self.rng_seed
        )

        self.register_buffer("rff_matrix", W)
        self.register_buffer(
            "random_offset",
            torch.rand(self.num_random_features, generator=random_generator)
            * 2
            * np.pi,
        )

        _num_repeats = ceil(self.num_random_features / self.length_scale_num)
        _length_scale = torch.repeat_interleave(
            torch.linspace(
                self.length_scale_low, self.length_scale_high, self.length_scale_num
            ),
            _num_repeats,
        )
        self.register_buffer("length_scale", _length_scale[: self.num_random_features])

        self.use_offset = use_offset
        if not self.use_offset:
            self.total_random_features = self.total_random_features * 2

    def feature_map(
        self,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the random-feature map for a given configuration :code:`h`

        Args:
            h (torch.Tensor):
                descriptors for a single configuration ~[natoms, descriptors]
            atomic_numbers (torch.Tensor):
                atomic numbers for a single configuration ~[natoms]
        """
        # Convert dtype of rf matrices to same as h
        self.rff_matrix = self.rff_matrix.to(dtype=h.dtype)
        if self.use_offset:
            self.random_offset = self.random_offset.to(dtype=h.dtype)

        self.length_scale = self.length_scale.to(dtype=h.dtype)
        length_scale = self.length_scale.view(-1, 1)
        # Compute the orthogonal random features nonlinear mapping
        Z = torch.einsum("af, Ff -> aF", h, self.rff_matrix / length_scale)
        if self.use_offset:
            b = self.random_offset.unsqueeze(0)
            Z = torch.cos(Z + b) / sqrt(0.5 * self.num_random_features)
        else:
            _s = torch.sin(Z)
            _c = torch.cos(Z)
            Z = torch.cat((_s, _c), dim=-1) / sqrt(self.num_random_features)

        return self.species_scatter_sum(
            Z, atomic_numbers=atomic_numbers
        )  # Sum over the single species

    def init_args(self):
        return super().init_args() | {
            "use_offset": self.use_offset,
            "length_scale_low": self.length_scale_low,
            "length_scale_high": self.length_scale_high,
            "length_scale_num": self.length_scale_num,
            "rng_seed": self.rng_seed,
        }


class Linear(RandomFeaturesHead):
    r"""Implements the linear kernel.

    This kernel does not need to be approximated with random-features, and can be used directly
    as if it were composed of random features. No stochasticity is present.

    .. math::

        \frac{\langle x, y \rangle}{\ell^{2}} + b

    .
    """

    def __init__(
        self,
        input_dim: int,
        num_species: int | None = None,
        chemically_informed_ratio: float | None = None,
        bias: float = 0.0,
        length_scale: float = 1.0,
    ):
        r"""Initialize a linear kernel with given bias :math:`b` and length scale :math:`\ell`.

        Args:
            input_dim (int): The dimensionality of the input features.
            num_species (int | None): The number of chemical species for
                which the kernel is computed. This parameter is relevant for systems
                with multiple chemical species. Defaults to :code:`None`.
            chemically_informed_ratio (float | None): The relative weight of chemically-informed kernels with respect to the all-species kernel. Ignored if
                :code:`num_species` is None. Defaults to :code:`None`.
            bias (float): The bias term :math:`b` added to the kernel function. This allows for shifting the kernel values, which can be useful for certain applications. Defaults to 0.0.
            length_scale (float): The length scale parameter :math:`\ell` that controls the smoothness of the kernel function. It affects how quickly the kernel values decay with distance. Defaults to 1.0.
        """
        check_positive_arg(length_scale, "length_scale")
        super(Linear, self).__init__(
            input_dim,
            # Add a constant bias feature
            input_dim + 1,
            num_species=num_species,
            chemically_informed_ratio=chemically_informed_ratio,
        )

        self.register_buffer("bias", torch.tensor(bias))
        self.register_buffer("length_scale", torch.tensor(length_scale))

    def feature_map(
        self,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the random-feature map for a given configuration :code:`h`

        Args:
            h (torch.Tensor):
                descriptors for a single configuration ~[natoms, descriptors]
            atomic_numbers (torch.Tensor):
                atomic numbers for a single configuration ~[natoms]
        """
        length_scale = self.length_scale
        scaled_descriptors = h / length_scale
        # Add a constant bias feature to scaled_descriptors
        bias_feature = (
            torch.sqrt(self.bias)
            .expand(scaled_descriptors.size(0), 1)
            .to(dtype=h.dtype)
        )
        scaled_descriptors = torch.cat([scaled_descriptors, bias_feature], dim=1)

        return self.species_scatter_sum(
            scaled_descriptors, atomic_numbers=atomic_numbers
        )

    def init_args(self):
        _super_init = super().init_args()
        _super_init.pop("num_random_features")
        return _super_init | {
            "bias": self.bias.tolist(),
            "length_scale": self.length_scale.tolist(),
        }


class TensorSketch(RandomFeaturesHead):
    r"""TensorSketch random-features approximation of the polynomial kernel

    .. math::
        \left(\frac{\langle x, y \rangle}{\ell^{2}} + b\right)^{d}


    from `"Fast and scalable polynomial kernels via explicit feature maps" <https://doi.org/10.1145/2487575.2487591>`_ by Pham and Pagh.
    """

    def __init__(
        self,
        input_dim: int,
        num_random_features: int = 2**10,
        num_species: int | None = None,
        chemically_informed_ratio: float | None = None,
        degree=2,
        bias: float = 0.0,
        length_scale: float = 1.0,
        rng_seed=None,
    ):
        r"""Initialize an approximation to the polynomial kernel of specified degree.

        Args:
            input_dim (int): The dimensionality of the input features.
            num_random_features (int): The number of random features to use in the sketching process. Defaults to :math:`2^{10} = 1024`.
            num_species (int | None): The number of chemical species for
                which the kernel is computed. This parameter is relevant for systems
                with multiple chemical species. Defaults to :code:`None`.
            chemically_informed_ratio (float | None): The relative weight of chemically-informed kernels with respect to the all-species kernel. Ignored if
                :code:`num_species` is None. Defaults to :code:`None`.
            degree (int): Degree :math:`d` of the polynomial kernel to sketch. Defaults to 2.
            bias (float): The bias term :math:`b` added to the kernel function. This allows for shifting the kernel values, which can be useful for certain applications. Defaults to 0.0.
            length_scale (float): The length scale parameter :math:`\ell` that controls the smoothness of the kernel function. It affects how quickly the kernel values decay with distance. Defaults to 1.0.
            rng_seed (int | None): A seed for the random number generator used in generating random features. Setting this ensures reproducibility of results. Defaults to :code:`None`.
        """
        check_positive_arg(length_scale, "length_scale")
        super(TensorSketch, self).__init__(
            input_dim,
            num_random_features,
            num_species=num_species,
            chemically_informed_ratio=chemically_informed_ratio,
        )
        self.rng_seed = rng_seed
        self.register_buffer("degree", torch.tensor(degree))
        self.register_buffer("bias", torch.tensor(bias))
        self.register_buffer("length_scale", torch.tensor(length_scale))
        self.register_buffer(
            "count_sketches", self._sample_count_sketches(self.rng_seed, self.degree)
        )

    def _sample_count_sketches(self, rng_seed, degree: int):
        rng_torch = get_seeded_generator(rng_seed)
        with torch.no_grad():
            count_sketches = []
            for _ in range(degree):
                count_sketches.append(self._sample_hash_fns(rng_torch))
            return torch.stack(count_sketches)  # ~[degree, random_features, input_dim]

    def _sample_hash_fns(self, generator: torch.Generator | None):
        i_hash = torch.randint(
            low=0,
            high=self.num_random_features,
            size=(self.input_dim,),
            generator=generator,
        )
        s_hash = torch.bernoulli(
            0.5 * torch.ones(self.input_dim), generator=generator
        )  # Bernoulli
        # Convert to Rademacher
        s_hash = s_hash * 2 - 1
        indices = torch.stack([i_hash, torch.arange(self.input_dim)], dim=0)
        count_sketch = torch.sparse_coo_tensor(
            indices,
            s_hash,
            torch.Size([self.num_random_features, self.input_dim]),
        ).to_dense()
        return count_sketch

    def feature_map(
        self,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the random-feature map for a given configuration :code:`h`

        Args:
            h (torch.Tensor):
                descriptors for a single configuration ~[natoms, descriptors]
            atomic_numbers (torch.Tensor):
                atomic numbers for a single configuration ~[natoms]
        """
        length_scale = self.length_scale
        scaled_descriptors = (h + torch.sqrt(torch.abs(self.bias))) / length_scale
        # Convert dtype of self to the same dtype as h.
        # Note: cannot use self.to(dtype) due to self.degree buffer being int!
        self.count_sketches = self.count_sketches.to(dtype=h.dtype)
        Z = torch.einsum(
            "af, dFf -> adF", scaled_descriptors, self.count_sketches
        )  # ~[atoms, degree, random_features]
        Z = torch.fft.rfft(Z, n=self.num_random_features)
        Z = Z.prod(1)  # ~[atoms, random_features]
        Z = torch.fft.irfft(Z, n=self.num_random_features)
        return self.species_scatter_sum(Z, atomic_numbers=atomic_numbers)

    def init_args(self):
        return super().init_args() | {
            "degree": self.degree.tolist(),
            "bias": self.bias.tolist(),
            "length_scale": self.length_scale.tolist(),
            "rng_seed": self.rng_seed,
        }


def check_positive_arg(value, name):
    if value <= 0:
        raise ValueError(f"Argument {name} must be positive. Got {value}.")


def get_seeded_generator(seed: int | None) -> torch.Generator | None:
    rng_torch = None
    if seed is not None:
        rng_torch = torch.Generator(device=torch.get_default_device())
        rng_torch.manual_seed(seed)
    return rng_torch


def initialize_rf(rf_config: RFConfig, rf_feature_dim: int):
    if isinstance(rf_config, GaussianRFConfig):
        rf_cls = OrthogonalRFF
    elif isinstance(rf_config, MultiscaleGaussianRFConfig):
        rf_cls = MultiScaleOrthogonalRFF
    else:
        raise ValueError(type(rf_config))
    random_features_params = sanitize_init_dict(rf_cls, dataclasses.asdict(rf_config))
    return rf_cls(input_dim=rf_feature_dim, **random_features_params)
