"""Combined LES architecture and optimization configuration."""

from franken.autotune.cli import parse_cli
from franken.config import LESConfig


def test_combined_config_checkpoint_roundtrip():
    config = LESConfig(
        hidden_dim=(64, 32),
        sigma=0.8,
        is_periodic=True,
        mode="alternating",
        optimizer="adam",
        batch_size=4,
        num_cycles=7,
        restore_best=False,
    )
    assert LESConfig.from_ckpt(config.to_ckpt()) == config


def test_architecture_only_checkpoint_uses_training_defaults():
    config = LESConfig.from_ckpt({"hidden_dim": (64, 32), "sigma": 0.8})
    assert config.hidden_dim == (64, 32)
    assert config.optimizer == "lbfgs"
    assert config.mode == "variable_projection"


def test_cli_combines_architecture_and_training():
    config = parse_cli(
        [
            "--train-path",
            "/tmp/train.xyz",
            "--backbone",
            "mace",
            "--mace.path-or-id",
            "mace_mp/small",
            "--rf",
            "gaussian",
            "--gaussian.num-rf",
            "8",
            "--les",
            "--les-sigma",
            "0.8",
            "--les-hidden-dim",
            "(64, 32)",
            "--les-dl",
            "2.5",
            "--les-n-max",
            "12",
            "--les-activation",
            "relu",
            "--les-num-cycles",
            "7",
            "--les-no-restore-best",
        ]
    )
    assert config.les.sigma == 0.8
    assert config.les.hidden_dim == (64, 32)
    assert config.les.dl == 2.5
    assert config.les.N_max == 12
    assert config.les.activation == "relu"
    assert config.les.num_cycles == 7
    assert config.les.restore_best is False
    assert not hasattr(config, "les_training")


def test_non_les_cli_does_not_build_les_config():
    config = parse_cli(
        [
            "--train-path",
            "/tmp/train.xyz",
            "--backbone",
            "mace",
            "--mace.path-or-id",
            "mace_mp/small",
            "--rf",
            "gaussian",
            "--gaussian.num-rf",
            "8",
        ]
    )
    assert config.les is None
