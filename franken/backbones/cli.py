import argparse

from franken.backbones import REGISTRY
from franken.backbones.utils import download_checkpoint, make_summary
from franken.utils.misc import setup_logger


### Command 'list': list available models


def build_list_arg_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("list", description="List available models")
    parser.add_argument(
        "--cache_dir",
        help=(
            "Directory to save the downloaded checkpoints. "
            "Defaults to '~/.franken/' in the user home or to the "
            "'FRANKEN_CACHE_DIR' environment variable if set."
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--log_level",
        help="log-level for the command-line logger",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def run_list_cmd(args):
    setup_logger(level=args.log_level, directory=None)
    print(make_summary(cache_dir=args.cache_dir))


### Command 'download': download a model


def build_download_arg_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("download", description="Download a model")
    parser.add_argument(
        "--model_name",
        help="The name of the model to download.",
        type=str,
        required=True,
        choices=[
            name for name, info in REGISTRY.items() if info["implemented"] is True
        ],
    )
    parser.add_argument(
        "--cache_dir",
        help=(
            "Directory to save the downloaded checkpoints. "
            "Defaults to '~/.franken/' in the user home or to the "
            "'FRANKEN_CACHE_DIR' environment variable if set."
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--log_level",
        help="log-level for the command-line logger",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def run_download_cmd(args):
    setup_logger(level=args.log_level, directory=None)
    download_checkpoint(args.model_name, args.cache_dir)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="List and download GNN backbones for franken."
    )

    subparsers = parser.add_subparsers(
        required=True,
        title="Franken backbone CLI",
        description="Provides helpers to interact with the various backbone models supported by Franken",
        help="Run `%(prog)s <subcommand> -h` for help with the individual subcommands",
    )

    list_parser = build_list_arg_parser(subparsers)
    list_parser.set_defaults(func=run_list_cmd)
    download_parser = build_download_arg_parser(subparsers)
    download_parser.set_defaults(func=run_download_cmd)

    return parser


def main():
    """This entry-point has 2 commands, 'list' and 'download'.
    Usage:
        franken.backbones list <args>
        franken.backbones download <args>
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


# For sphinx docs
get_parser_fn = lambda: build_arg_parser()  # noqa: E731
