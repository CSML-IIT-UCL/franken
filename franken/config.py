from abc import ABC
import ast
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, Literal, Sequence, Union

import tyro


def all_fields(class_or_instance):
    """Return a tuple describing the fields of this dataclass.

    Accepts a dataclass or an instance of one. Tuple elements are of
    type Field.
    """

    try:
        fields = getattr(class_or_instance, "__dataclass_fields__")
    except AttributeError:
        raise TypeError("must be called with a dataclass type or instance") from None

    # Exclude pseudo-fields.  Note that fields is sorted by insertion
    # order, so the order of the tuple is as the fields were defined.
    return tuple(
        f
        for f in fields.values()
        if f._field_type.name == "_FIELD_CLASSVAR" or f._field_type.name == "_FIELD"
    )


def asdict_with_classvar(obj) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        result = []
        for f in all_fields(obj):
            value = asdict_with_classvar(getattr(obj, f.name))
            result.append((f.name, value))
        return dict(result)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*[asdict_with_classvar(v) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(asdict_with_classvar(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (asdict_with_classvar(k), asdict_with_classvar(v)) for k, v in obj.items()
        )
    else:
        return deepcopy(obj)


@dataclass
class HPSearchConfig:
    value: float | None = None
    values: Sequence[float] | None = None
    start: float | None = None
    stop: float | None = None
    num: int | None = None
    scale: Literal["log", "linear"] | None = None

    def get_vals(self) -> list[float | int]:
        if self.value is not None:
            return [self.value]
        elif self.values is not None:
            return list(self.values)
        else:
            assert (
                self.start is not None
                and self.stop is not None
                and self.num is not None
                and self.scale is not None
            )
            import numpy as np

            if self.scale == "log":
                return np.logspace(self.start, self.stop, self.num).tolist()
            elif self.scale == "linear":
                return np.linspace(self.start, self.stop, self.num).tolist()
            else:
                raise ValueError(self.scale)

    @staticmethod
    def from_str(hp_str: str) -> "HPSearchConfig":
        try:
            val = float(hp_str)
            return HPSearchConfig(value=val)
        except ValueError:
            pass
        try:
            vals = ast.literal_eval(hp_str)
            if isinstance(vals, (tuple, list)):
                if all(isinstance(v, (int, float)) for v in vals):
                    return HPSearchConfig(values=vals)
                else:
                    if (
                        len(vals) == 4
                        and vals[-1] in {"log", "linear"}
                        and all(isinstance(v, (int, float)) for v in vals[:3])
                    ):
                        return HPSearchConfig(
                            start=vals[0], stop=vals[1], num=vals[2], scale=vals[3]
                        )
        except Exception:
            pass
        raise TypeError(
            f"String '{hp_str}' cannot be converted to an instance of HPSearchConfig. You can either provide float or int values, "
            f"lists of float or int values, or a 4-tuple `(<start>, <stop>, <num>, <'log' | 'linear'>)` which specifies a linear "
            f"or logarithmic range."
        )

    @classmethod
    def is_string_valid(cls, hp_str: str) -> bool:
        try:
            cls.from_str(hp_str)
            return True
        except Exception:
            return False

    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.value}"
        elif self.values is not None:
            return f"{self.values}"
        else:
            return f"({self.start}, {self.stop}, {self.num}, {self.scale})"

    def __post_init__(self):
        if self.value is None:
            if self.values is None:
                if (
                    self.start is None
                    or self.stop is None
                    or self.num is None
                    or self.scale not in {"log", "linear"}
                ):
                    raise TypeError("Invalid HPSearchConfig object was created.")


HPSearchConfigStringType = Annotated[
    HPSearchConfig,
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="HYPERPARAMETER",
        instance_from_str=lambda args: HPSearchConfig.from_str(args[0]),
        is_instance=HPSearchConfig.is_string_valid,
        str_from_instance=lambda instance: [str(instance)],
    ),
]


@dataclass
class DatasetConfig:
    name: str
    """Dataset name. Can be either one of the predefined datasets, or a custom one."""

    train_path: str | None = None
    """Path to the training dataset. This should be readable with `ase`, e.g. xyz files work."""

    test_path: str | None = None
    """Path to the test dataset. When doing hyperparameter tuning this
    is never used to determine the best model."""

    val_path: str | None = None
    """Path to the validation dataset. Used for hyperparameter tuning if available."""

    max_train_samples: int | None = None
    """Randomly subsample the training set to have at most `max_train_samples` points."""


@dataclass
class BackboneConfig(ABC):
    path_or_id: str
    """Path to the GNN checkpoint, or an ID corresponding to a predefined checkpoint.
    To list the available IDs please run the `franken.backbones list` command."""

    interaction_block: int
    """GNN layer out of which the features are extracted."""

    family: ClassVar[str]

    def to_ckpt(self):
        return asdict_with_classvar(self)

    @staticmethod
    def from_ckpt(ckpt):
        if ckpt["family"] == "mace":
            cls = MaceBackboneConfig
        elif ckpt["family"] == "fairchem":
            cls = FairchemBackboneConfig
        elif ckpt["family"] == "sevenn":
            cls = SevennBackboneConfig
        else:
            raise ValueError(ckpt["family"])
        init_args = deepcopy(ckpt)
        init_args.pop("family")
        return cls(**init_args)


@dataclass
class MaceBackboneConfig(BackboneConfig):
    family: ClassVar[str] = "mace"
    interaction_block: int = 2
    """GNN layer out of which the features are extracted."""


@dataclass
class FairchemBackboneConfig(BackboneConfig):
    family: ClassVar[str] = "fairchem"
    interaction_block: int = 2
    """GNN layer out of which the features are extracted."""


@dataclass
class SevennBackboneConfig(BackboneConfig):
    family: ClassVar[str] = "sevenn"
    interaction_block: int = 2
    """GNN layer out of which the features are extracted."""

    extract_after_act: bool = True
    """Whether features should be extracted before or after activations."""

    append_layers: bool = True
    """Whether to take only the features from the last interaction layer, or to concatenate them all."""


@dataclass
class RFConfig(ABC):
    rf_type: ClassVar[str]

    def to_ckpt(self):
        return asdict_with_classvar(self)

    @staticmethod
    def from_ckpt(ckpt):
        if ckpt["rf_type"] == "gaussian":
            cls = GaussianRFConfig
        elif ckpt["rf_type"] == "multiscale-gaussian":
            cls = MultiscaleGaussianRFConfig
        else:
            raise ValueError(ckpt["rf_type"])
        init_args = deepcopy(ckpt)
        init_args.pop("rf_type")
        return cls(**init_args)


@dataclass
class GaussianRFConfig(RFConfig):
    rf_type: ClassVar[str] = "gaussian"
    num_random_features: int
    """Number of random features"""

    length_scale: tyro.conf.AvoidSubcommands[
        HPSearchConfigStringType | list[float] | float
    ] = field(
        default_factory=lambda: HPSearchConfig(start=4, stop=20, num=10, scale="linear")
    )
    """Gaussian length-scale sigma."""

    use_offset: bool = True
    """Whether or not to use the version of Gaussian random features with a uniform offset. Set to true, increases the estimator's efficiency."""

    rng_seed: int = 1337
    """Random seed for reproducibility of the random features"""


@dataclass
class MultiscaleGaussianRFConfig(RFConfig):
    rf_type: ClassVar[str] = "multiscale-gaussian"
    num_random_features: int
    """Number of random features"""

    length_scale_low: float = 4
    """Bottom of the range for the length-scale. Should be positive."""

    length_scale_high: float = 20
    """Top of the range for the length-scale. Should be positive."""

    length_scale_num: int = 6
    """Number of different length-scales to include within the specified range."""

    use_offset: bool = True
    """Whether or not to use the version of Gaussian random features with a uniform offset. Set to true, increases the estimator's efficiency."""

    rng_seed: int = 1337
    """Random seed for reproducibility of the random features"""


@dataclass
class SolverConfig:
    l2_penalty: tyro.conf.AvoidSubcommands[
        HPSearchConfigStringType | list[float] | float
    ] = field(
        default_factory=lambda: HPSearchConfig(start=-11, stop=-6, num=6, scale="log")
    )
    """The amount of regularization. Should be a small positive number."""

    force_weight: tyro.conf.AvoidSubcommands[
        HPSearchConfigStringType | list[float] | float
    ] = field(
        default_factory=lambda: HPSearchConfig(
            start=0.01, stop=0.99, num=10, scale="linear"
        )
    )
    """Controls how much weight the forces term, as opposed to the energy term has in the loss. Should be a number between 0 and 1."""


@dataclass
class AutotuneConfig:
    """Configure automatic hyperparameter tuning for franken.

    The program will run a grid-search over certain hyperparameters (denoted by 'HYPERPARAMETER' in the help-text), which can be configured by using the following three forms:
    1. a simple float or int value,
    2. a list of float or int values,
    3. a 4-tuple `(<start>, <stop>, <num>, <'log' | 'linear'>)` which specifies a linear or logarithmic range.

    There are two classes of hyperparameters over which we can perform grid-search: solver parameters (for example the L2 penalty), and random feature parameters. It is much more efficient to try multiple combinations of the solver parameters than of the random feature parameters, so plan your grid accordingly.
    In particular, some random feature approximations like the `multiscale-gaussian` do not need any hyperparameter search, reducing the time required for the overall grid-search.

    The program has two sub-commands: `backbone` which allows to choose the GNN underlying the franken features, and `rfs` to choose the random feature approximation and its parameters.
    To view the help-text for the subcommands, you can run for example
    `franken.autotune backbone:mace -h`
    or
    `franken.autotune backbone:mace rfs:gaussian -h`
    Note that the backbone must be chosen before the random features
    """

    dataset: DatasetConfig
    """Configure a dataset for training Franken.

    If `--dataset.name` corresponds to one of the datasets used in the Franken paper (e.g. "water", "PtH2O", "TM23/Ag", etc.) there is no need to specify train, test or validation paths: the code will take care of downloading and preprocessing the data automatically.
    Instead, to use a custom dataset please specify at a minimum the training path, and ideally also the validation path (which is used to determine the best model during a hyperparameter search).
    """

    solver: SolverConfig
    """Configure the franken solver. Hyperparameter search over these is efficient, so the search-grid can be quite fine-grained."""

    backbone: Union[
        Annotated[
            MaceBackboneConfig,
            tyro.conf.subcommand(
                name="mace", description="Configure the MACE GNN backbone."
            ),
        ],
        Annotated[
            FairchemBackboneConfig,
            tyro.conf.subcommand(
                name="fairchem", description="Configure the Fairchem GNN backbone."
            ),
        ],
        Annotated[
            SevennBackboneConfig,
            tyro.conf.subcommand(
                name="sevenn", description="Configure the SevenNet GNN backbone."
            ),
        ],
    ]
    """The GNN backbone which will be used by franken."""

    rfs: Union[
        Annotated[
            GaussianRFConfig,
            tyro.conf.subcommand(
                name="gaussian",
                description="Orthogonal random features to approximate the Gaussian kernel.",
            ),
        ],
        Annotated[
            MultiscaleGaussianRFConfig,
            tyro.conf.subcommand(
                name="multiscale-gaussian",
                description="Random features to approximate a Gaussian kernel with multiple length-scales.",
            ),
        ],
    ]
    """Choose the random-feature approximation."""

    rf_normalization: Literal["leading_eig"] | None = "leading_eig"
    """Normalization strategy for the covariance matrix."""

    save_every_model: bool = False
    """If true saves a checkpoint for every trial, otherwise it saves only the best model."""

    dtype: Literal["float32", "float64"] = "float64"
    """Data-type for the franken solution. float64 can usually obtain slightly smaller errors while paying a small performance cost."""

    save_fmaps: bool = False
    """Whether to save training feature maps. If the dataset is small (~100 samples), setting this to True can increase the speed of hyperparameter tuning, at the cost of higher memory usage."""

    scale_by_species: bool = True
    """how to scale the GNN features, whether globally (across species) or individually per species."""

    jac_chunk_size: Union[Literal["auto",], int] = "auto"
    """Chunk-size for jacobian calculations. 'auto' attempts to set it based on available GPU memory. If you encounter out-of-memory issues, try setting this manually."""

    run_dir: str = "."
    """Directory in which the hyperparameter search results will be saved"""

    seed: int = 1337
    """Random seed"""

    console_logging_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    """Controls verbosity"""
