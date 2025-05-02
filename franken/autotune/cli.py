import argparse
import dataclasses
import types
import typing
import docstring_parser
from typing import Any, Sequence

from franken.config import (
    AutotuneConfig,
    MaceBackboneConfig,
    FairchemBackboneConfig,
    SevennBackboneConfig,
    GaussianRFConfig,
    MultiscaleGaussianRFConfig,
    SolverConfig,
    DatasetConfig,
    HPSearchConfig,
)


def parse_docstring_from_object(obj: object) -> dict[str, str]:
    return {
        doc.arg_name: doc.description
        for doc in docstring_parser.parse_from_object(obj).params
        if doc.description is not None
    }


def get_field_docstring(cls: typing.Type, field_name: str):
    """Taken from tyro https://github.com/brentyi/tyro/blob/main/src/tyro/_docstrings.py#L25"""
    # NoneType will break docstring_parser.
    if cls is type(None):
        return None

    # Try to parse using docstring_parser.
    for cls_search in cls.__mro__:
        if cls_search.__module__ == "builtins":
            continue  # Skip `object`, `Callable`, `tuple`, etc.
        docstring = parse_docstring_from_object(cls_search).get(field_name, None)
        if docstring is not None:
            return docstring.strip()


def field_is_optional(field_type: Any):
    """A field is optional when it has Union type with a NoneType alternative.
    Note that Optional[] is a special form which is converted to a Union with a NoneType option
    """
    origin = typing.get_origin(field_type)
    if origin is typing.Union:
        return type(None) in typing.get_args(field_type)
    if origin is types.UnionType:
        return type(None) in typing.get_args(field_type)
    return False


def field_is_optional_str(field_type: Any):
    if not field_is_optional(field_type):
        return False
    is_str_present = type(str) in typing.get_args(field_type)
    return is_str_present and len(typing.get_args(field_type)) == 2


def field_is_optional_literal(field_type: Any):
    if not field_is_optional(field_type):
        return False
    all_subtypes = typing.get_args(field_type)
    # To check for literal we must check the origin of the sub-types
    all_subtype_origins = [typing.get_origin(t) for t in all_subtypes]
    is_lit_present = typing.Literal in all_subtype_origins
    return is_lit_present and len(all_subtypes) == 2


def parse_optional_literal(s: str) -> str | None:
    if s == "none":
        return None
    return s


class Argument:
    def __init__(self, full_name: str, dataclass, dataclass_name: str, **kwargs):
        self.full_name = full_name
        self.dataclass = dataclass
        self.dataclass_name = dataclass_name
        self.help = get_field_docstring(dataclass, dataclass_name)
        self.arg_config = kwargs
        self.is_arg_required = "default" not in self.arg_config
        self.dest = None

    def add(self, arg_container, **kwargs):
        try:
            action = arg_container.add_argument(
                f"--{self.full_name}", help=self.help, **kwargs
            )
        except TypeError as e:
            raise RuntimeError(
                f"Failed to add argument {self} with arguments {kwargs}."
            ) from e
        self.dest = action.dest

    def is_added(self) -> bool:
        return self.dest is not None

    @staticmethod
    def from_dataclass(
        dataclass, dataclass_name, full_name, opposite_full_name=None, **kwargs
    ):
        all_fields = dataclasses.fields(dataclass)
        try:
            named_field = [f for f in all_fields if f.name == dataclass_name][0]
        except IndexError:
            raise ValueError(
                f"Dataclass {dataclass} does not have attribute {dataclass_name}"
            )
        arg_kwargs = {}

        if named_field.default != dataclasses.MISSING:
            arg_kwargs["default"] = str(named_field.default)
        elif named_field.default_factory != dataclasses.MISSING:
            arg_kwargs["default"] = str(named_field.default_factory())

        if named_field.type in {int, float, str}:
            arg_kwargs["type"] = named_field.type

        # Literals
        type_origin = typing.get_origin(named_field.type)
        if type_origin == typing.Literal:
            arg_kwargs["choices"] = typing.get_args(named_field.type)
        if field_is_optional_literal(named_field.type):
            arg_kwargs["choices"] = list(typing.get_args(named_field.type)) + ["none"]
            arg_kwargs["type"] = parse_optional_literal

        # Booleans
        if named_field.type is bool:
            if named_field.default is False:
                arg_kwargs["action"] = "store_true"
            elif named_field.default is True:
                arg_kwargs["action"] = "store_false"
                if opposite_full_name is None:
                    raise ValueError(
                        f"opposite_full_name must be non-null for field {dataclass_name}"
                    )
                full_name = opposite_full_name

        return Argument(
            full_name=full_name,
            dataclass=dataclass,
            dataclass_name=dataclass_name,
            **(arg_kwargs | kwargs),
        )

    def __repr__(self):
        return f"Argument({self.full_name=} {self.dataclass=} {self.dataclass_name=})"


class ArgumentGroup:
    def __init__(
        self,
        name: str,
        title: str,
        desc: str,
        data_class: type,
        arguments: Sequence[Argument],
    ):
        self.name = name
        self.title = title
        self.desc = desc
        self.data_class = data_class
        self.arguments = arguments

    def add_to_parser(self, parser: argparse.ArgumentParser):
        arg_group = parser.add_argument_group(
            title=self.title, description=self.desc if self.desc else None
        )
        for arg in self.arguments:
            arg.add(arg_group, **(arg.arg_config | {"required": arg.is_arg_required}))

    def to_dataclass(self, args):
        dc_args = {}
        for arg_config in self.arguments:
            if arg_config.dest is None:
                raise RuntimeError(
                    f"dest attribute for argument {arg_config.full_name} is "
                    f"None. Have you parsed arguments correctly?"
                )
            dc_args[arg_config.dataclass_name] = getattr(args, arg_config.dest)
        return self.data_class(**dc_args)


class MutuallyExclusiveArgumentGroup(ArgumentGroup):
    def __init__(self, name: str, title: str, desc: str, data_class: type, arguments):
        for arg in arguments:
            arg.full_name = f"{name}.{arg.full_name}"  # e.g. fairchem.interaction_block
        super().__init__(name, title, desc, data_class, arguments)

    def add_to_parser(self, parser: argparse.ArgumentParser):
        arg_group = parser.add_argument_group(
            title=self.title, description=self.desc if self.desc else None
        )
        for arg in self.arguments:
            arg.add(arg_group, **arg.arg_config)


class GroupWithIndividualOptions:
    def __init__(
        self,
        group_name: str,
        me_groups: Sequence[MutuallyExclusiveArgumentGroup],
        help_text: str,
    ):
        self.group_name = group_name
        self.me_groups = {g.name: g for g in me_groups}
        self.help_text = help_text

    def add_to_parser(self, parser):
        parser.add_argument(
            f"--{self.group_name}",
            choices=list(self.me_groups.keys()),
            required=True,
            help=self.help_text,
        )
        for g_name, g in self.me_groups.items():
            g.add_to_parser(parser)

    def get_selected_subgroup(self, args) -> tuple[MutuallyExclusiveArgumentGroup, str]:
        parsed_subgroup_name = getattr(args, self.group_name, None)
        if parsed_subgroup_name is None:
            raise ValueError(
                f"Group {self.group_name} does not have any value in parsed arguments"
            )
        try:
            selected_subgroup = self.me_groups[parsed_subgroup_name]
        except KeyError:
            raise ValueError(
                f"Group {self.group_name} with value {parsed_subgroup_name} has no registered group data."
            )
        return selected_subgroup, parsed_subgroup_name

    def validate_required(self, parser, args):
        selected_subgroup, subgroup_name = self.get_selected_subgroup(args)

        for arg_config in selected_subgroup.arguments:
            if arg_config.dest is None:
                raise RuntimeError(
                    f"dest attribute for argument {arg_config.full_name} is "
                    f"None. Have you parsed arguments correctly?"
                )
            if arg_config.is_arg_required and getattr(args, arg_config.dest) is None:
                parser.error(
                    f"--{arg_config.full_name} is required when --{self.group_name}={subgroup_name}"
                )

    def to_dataclass(self, args):
        selected_subgroup, _ = self.get_selected_subgroup(args)
        return selected_subgroup.to_dataclass(args)


def get_args_from_dataclass(dataclass, full_name_prefix=""):
    args = []
    for f in dataclasses.fields(dataclass):
        full_name = f"{full_name_prefix}.{f.name.replace('_', '-')}"
        args.append(Argument.from_dataclass(dataclass, f.name, full_name))


def get_arg_groups():
    solver_arg_group = ArgumentGroup(
        "solver",
        "Solver",
        desc="",
        data_class=SolverConfig,
        arguments=[
            Argument.from_dataclass(
                SolverConfig,
                "l2_penalty",
                "l2-penalty",
                metavar="HYPERPARAMETER",
                type=HPSearchConfig.from_str,
            ),
            Argument.from_dataclass(
                SolverConfig,
                "force_weight",
                "force-weight",
                metavar="HYPERPARAMETER",
                type=HPSearchConfig.from_str,
            ),
        ],
    )
    dset_arg_group = ArgumentGroup(
        "dataset",
        "Dataset",
        desc="",
        data_class=DatasetConfig,
        arguments=[
            Argument.from_dataclass(DatasetConfig, "name", "dataset-name"),
            Argument.from_dataclass(DatasetConfig, "train_path", "train-path"),
            Argument.from_dataclass(DatasetConfig, "test_path", "test-path"),
            Argument.from_dataclass(DatasetConfig, "val_path", "val-path"),
            Argument.from_dataclass(
                DatasetConfig, "max_train_samples", "max-train-samples"
            ),
        ],
    )
    bbone_groups = GroupWithIndividualOptions(
        "backbone",
        [
            MutuallyExclusiveArgumentGroup(
                "mace",
                title="MACE backbone",
                desc="Configure the MACE backbone. Specify ``--backbone=mace`` to enable.",
                data_class=MaceBackboneConfig,
                arguments=[
                    Argument.from_dataclass(
                        MaceBackboneConfig, "path_or_id", "path-or-id"
                    ),
                    Argument.from_dataclass(
                        MaceBackboneConfig, "interaction_block", "interaction-block"
                    ),
                ],
            ),
            MutuallyExclusiveArgumentGroup(
                "sevenn",
                title="SevenNet backbone",
                desc="Configure the MACE backbone. Specify ``--backbone=sevenn`` to enable.",
                data_class=SevennBackboneConfig,
                arguments=[
                    Argument.from_dataclass(
                        SevennBackboneConfig, "path_or_id", "path-or-id"
                    ),
                    Argument.from_dataclass(
                        SevennBackboneConfig, "interaction_block", "interaction-block"
                    ),
                    Argument.from_dataclass(
                        SevennBackboneConfig,
                        "extract_after_act",
                        "extract-after-act",
                        "extract-before-act",
                    ),
                    Argument.from_dataclass(
                        SevennBackboneConfig,
                        "append_layers",
                        "append-layers",
                        "last-layer-only",
                    ),
                ],
            ),
            MutuallyExclusiveArgumentGroup(
                "fairchem",
                title="Fairchem backbone",
                desc="Configure the MACE backbone. Specify ``--backbone=fairchem`` to enable.",
                data_class=FairchemBackboneConfig,
                arguments=[
                    Argument.from_dataclass(
                        FairchemBackboneConfig, "path_or_id", "path-or-id"
                    ),
                    Argument.from_dataclass(
                        FairchemBackboneConfig, "interaction_block", "interaction-block"
                    ),
                ],
            ),
        ],
        help_text="The GNN backbone which will be used by franken.",
    )
    rf_groups = GroupWithIndividualOptions(
        "rf",
        [
            MutuallyExclusiveArgumentGroup(
                "gaussian",
                title="Gaussian RFs",
                desc="Configuration group for orthogonal random fourier features (ORFF), which are used to approximate a Gaussian kernel. Specify ``--rf=gaussian`` to enable this RF approximation",
                data_class=GaussianRFConfig,
                arguments=[
                    Argument.from_dataclass(
                        GaussianRFConfig, "num_random_features", "num-rf"
                    ),
                    Argument.from_dataclass(
                        GaussianRFConfig,
                        "length_scale",
                        "length-scale",
                        metavar="HYPERPARAMETER",
                        type=HPSearchConfig.from_str,
                    ),
                    Argument.from_dataclass(
                        GaussianRFConfig, "use_offset", "use-offset", "no-use-offset"
                    ),
                    Argument.from_dataclass(GaussianRFConfig, "rng_seed", "rng-seed"),
                ],
            ),
            MutuallyExclusiveArgumentGroup(
                "ms-gaussian",
                title="Multi-scale Gaussian RFs",
                desc="Configuration group for multi-scale orthogonal random fourier features (ORFF), which are used to approximate a Gaussian kernel with multiple length-scales. Specify ``--rf=ms-gaussian`` to enable this RF approximation",
                data_class=MultiscaleGaussianRFConfig,
                arguments=[
                    Argument.from_dataclass(
                        MultiscaleGaussianRFConfig, "num_random_features", "num-rf"
                    ),
                    Argument.from_dataclass(
                        MultiscaleGaussianRFConfig,
                        "use_offset",
                        "use-offset",
                        "no-use-offset",
                    ),
                    Argument.from_dataclass(
                        MultiscaleGaussianRFConfig,
                        "length_scale_low",
                        "length-scale-low",
                    ),
                    Argument.from_dataclass(
                        MultiscaleGaussianRFConfig,
                        "length_scale_high",
                        "length-scale-high",
                    ),
                    Argument.from_dataclass(
                        MultiscaleGaussianRFConfig,
                        "length_scale_num",
                        "length-scale-num",
                    ),
                    Argument.from_dataclass(
                        MultiscaleGaussianRFConfig, "rng_seed", "rng-seed"
                    ),
                ],
            ),
        ],
        help_text="Choose the random-feature approximation.",
    )
    return {
        "solver": solver_arg_group,
        "dataset": dset_arg_group,
        "backbone": bbone_groups,
        "rfs": rf_groups,
    }


def build_parser(return_groups: bool = False):
    parser = argparse.ArgumentParser(
        prog="franken.autotune",
        description="Franken autotune: automatic hyperparameter tuning for franken models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--rf-norm",
        choices=["leading_eig", "none"],
        default="leading_eig",
        help=get_field_docstring(AutotuneConfig, "rf_normalization"),
    )
    parser.add_argument(
        "--save-every-model",
        action="store_true",
        help=get_field_docstring(AutotuneConfig, "save_every_model"),
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help=get_field_docstring(AutotuneConfig, "dtype"),
    )
    parser.add_argument(
        "--save-fmaps",
        action="store_true",
        help=get_field_docstring(AutotuneConfig, "save_fmaps"),
    )
    parser.add_argument(
        "--global-scaling",
        action="store_false",
        help=get_field_docstring(AutotuneConfig, "scale_by_species"),
    )
    parser.add_argument(
        "--jac-chunk-size",
        default="auto",
        help=get_field_docstring(AutotuneConfig, "jac_chunk_size"),
    )
    parser.add_argument(
        "--run-dir",
        "-p",
        type=str,
        default=".",
        help=get_field_docstring(AutotuneConfig, "run_dir"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help=get_field_docstring(AutotuneConfig, "seed"),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="INFO",
        help=get_field_docstring(AutotuneConfig, "console_logging_level"),
    )

    arg_groups = get_arg_groups()
    for g in arg_groups.values():
        g.add_to_parser(parser)

    if return_groups:
        return parser, arg_groups
    return parser


def parse_cli(argv):
    parser, groups = build_parser(True)
    args = parser.parse_args(argv)

    groups["backbone"].validate_required(parser, args)
    groups["rfs"].validate_required(parser, args)

    bbone_config = groups["backbone"].to_dataclass(args)
    rf_config = groups["rfs"].to_dataclass(args)
    solver_config = groups["solver"].to_dataclass(args)
    dset_config = groups["dataset"].to_dataclass(args)

    # Initialize autotune config
    autotune_cfg = AutotuneConfig(
        dataset=dset_config,
        solver=solver_config,
        backbone=bbone_config,
        rfs=rf_config,
        rf_normalization=args.rf_norm,
        save_every_model=args.save_every_model,
        dtype=args.dtype,
        save_fmaps=args.save_fmaps,
        scale_by_species=not args.global_scaling,
        jac_chunk_size=args.jac_chunk_size,
        run_dir=args.run_dir,
        seed=args.seed,
        console_logging_level=args.log_level,
    )
    return autotune_cfg
