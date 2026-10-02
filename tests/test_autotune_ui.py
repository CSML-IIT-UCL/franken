"""Focused UI-generator checks; also runnable without the model-download fixtures.

    python -m unittest tests.test_autotune_ui
"""

import importlib.util
import json
from pathlib import Path
import unittest

from franken.autotune.cli import build_parser, parse_cli

spec = importlib.util.spec_from_file_location(
    "autotune_ui", Path(__file__).resolve().parents[1] / "scripts/autotune_ui.py"
)
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)
build_schema, generate_html, validate_argv = (
    ui.build_schema,
    ui.generate_html,
    ui.validate_argv,
)


class AutotuneUITest(unittest.TestCase):
    def test_every_cli_option_has_a_control(self):
        fields = build_schema()["fields"]
        actions = [a for a in build_parser()._actions if a.dest != "help"]
        self.assertEqual(
            {f["flag"] for f in fields}, {a.option_strings[0] for a in actions}
        )
        for field, action in zip(fields, actions):
            self.assertEqual(field["help"], action.help or "")
            self.assertEqual(
                field["choices"],
                list(action.choices) if action.choices is not None else None,
            )

    def test_readable_labels_retain_the_original_flags(self):
        fields = {f["flag"]: f for f in build_schema()["fields"]}
        self.assertEqual(fields["--train-path"]["label"], "Train Path")
        self.assertEqual(fields["--mace.path-or-id"]["label"], "Path Or ID")
        self.assertEqual(
            fields["--gaussian.num-rf"]["label"], "Number of Random Features"
        )
        for field in fields.values():
            self.assertFalse(field["label"].startswith("--"))
            self.assertTrue(field["flag"].startswith("--"))

    def test_conditional_requirements_and_inverse_flags(self):
        fields = {f["flag"]: f for f in build_schema()["fields"]}
        self.assertTrue(fields["--pet.path-or-id"]["required"])
        self.assertEqual(fields["--pet.path-or-id"]["condition"], "pet")
        self.assertEqual(fields["--ms-gaussian.num-rf"]["condition"], "ms-gaussian")
        for flag in (
            "--sevenn.last-layer-only",
            "--sevenn.extract-before-act",
            "--gaussian.no-use-offset",
        ):
            self.assertTrue(fields[flag]["default"])
            self.assertFalse(fields[flag]["const"])

    def test_parser_validation_for_each_model_and_rf(self):
        for backbone in ("mace", "pet", "sevenn"):
            for rf in ("gaussian", "ms-gaussian"):
                argv = [
                    "--train-path=/tmp/data with spaces.xyz",
                    "--backbone=" + backbone,
                    f"--{backbone}.path-or-id=checkpoint.pt",
                    "--rf=" + rf,
                    f"--{rf}.num-rf=128",
                ]
                with self.subTest(backbone=backbone, rf=rf):
                    self.assertTrue(validate_argv(argv)["valid"])
                    cfg = parse_cli(argv)
                    self.assertEqual(
                        cfg.dataset.train_path, "/tmp/data with spaces.xyz"
                    )
                    self.assertEqual(cfg.rfs.num_random_features, 128)

    def test_rejects_missing_and_malformed_options(self):
        base = [
            "--train-path=train.xyz",
            "--backbone=mace",
            "--mace.path-or-id=checkpoint.pt",
            "--rf=gaussian",
            "--gaussian.num-rf=128",
        ]
        for argv in (
            [],
            base[:-1],
            base + ["--l2-penalty=garbage"],
            base + ["--atomic-energies=[]"],
            base + ["--train-targets", "unknown"],
            base + ["--metrics", "unknown"],
            base + ["--seed=1.5"],
            ["--help"],
        ):
            with self.subTest(argv=argv):
                self.assertFalse(validate_argv(argv)["valid"])

    def test_hyperparameter_forms_and_atomic_energies(self):
        base = [
            "--dataset-name=test",
            "--backbone=sevenn",
            "--sevenn.path-or-id=checkpoint.pt",
            "--rf=gaussian",
            "--gaussian.num-rf=128",
            "--sevenn.last-layer-only",
            "--atomic-energies={1: -0.5, 8: -75.3}",
        ]
        for hp in ("1e-6", "[1e-6, 1e-4]", "(-6, -2, 5, log)", "(1, 4, 3, linear)"):
            self.assertTrue(validate_argv(base + ["--l2-penalty=" + hp])["valid"])
        self.assertFalse(parse_cli(base).backbone.append_layers)

    def test_embedded_schema_is_valid_json(self):
        html = generate_html()
        embedded = html.split('<script id="schema" type="application/json">')[1].split(
            "</script>"
        )[0]
        self.assertEqual(json.loads(embedded), build_schema())
        self.assertNotIn("__SCHEMA__", html)


if __name__ == "__main__":
    unittest.main()
