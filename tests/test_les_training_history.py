"""History logging checks without a backbone or training run."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from franken.trainers.log_utils import DataSplit, LogEntry
from franken.trainers.rf_ewalds import RandomFeaturesEwaldsTrainer
from franken.config import LESTrainingConfig


class TestLESTrainingHistory(unittest.TestCase):
    def make_trainer(self, log_dir):
        trainer = object.__new__(RandomFeaturesEwaldsTrainer)
        trainer.log_dir = log_dir
        trainer.training_history = []
        trainer.training_config = LESTrainingConfig(restore_best=False)
        trainer.train_dataloader = Mock(name="train_loader")
        trainer.val_dataloader = Mock(name="validation_loader")
        trainer.create_log_entry = Mock(
            side_effect=lambda *args: LogEntry("model123", 0, 0, 0)
        )
        trainer.eval_summary = Mock(return_value="evaluation")

        def evaluate(model, loader, log_collection, all_weights):
            split = (
                DataSplit.TRAIN
                if loader is trainer.train_dataloader
                else DataSplit.VALIDATION
            )
            for name, value in (("energy_RMSE", 1.5), ("forces_RMSE", 20.0)):
                log_collection[0].add_metric(name, value, split)
            return log_collection

        trainer.evaluate = Mock(side_effect=evaluate)
        return trainer

    @patch("franken.trainers.rf_ewalds.dist_utils.get_rank", return_value=0)
    def test_records_both_stages_and_splits_with_current_weights(self, rank):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp) / "run")
            model = Mock()
            weights = torch.tensor([[2.0]])
            path = Path(tmp) / "run" / "training_history.json"
            trainer._print_eval({}, model, weights, 0, step="rff")
            first = json.loads(path.read_text())
            self.assertEqual(len(first), 1)
            for call in trainer.evaluate.call_args_list:
                self.assertIs(call.kwargs["all_weights"], weights)
            trainer._print_eval({}, model, None, 0, step="les")
            trainer._print_eval({}, model, weights, 1, step="rff")
            saved = json.loads(path.read_text())
            self.assertEqual(saved, trainer.training_history)
            self.assertEqual(
                [(row["cycle"], row["step"]) for row in saved],
                [(1, "rff"), (1, "les"), (2, "rff")],
            )
            for row in saved:
                for split in ("train", "validation"):
                    self.assertEqual(row["metrics"][split]["energy_RMSE"], 1.5)
                    self.assertEqual(row["metrics"][split]["forces_RMSE"], 20.0)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    @patch("franken.trainers.rf_ewalds.dist_utils.get_rank", return_value=0)
    def test_without_validation_or_log_directory(self, rank):
        trainer = self.make_trainer(None)
        trainer.val_dataloader = None
        trainer._print_eval({}, Mock(), None, 0, step="joint")
        self.assertEqual(trainer.evaluate.call_count, 1)
        self.assertEqual(trainer.training_history[0]["step"], "joint")
        self.assertEqual(set(trainer.training_history[0]["metrics"]), {"train"})

    @patch("franken.trainers.rf_ewalds.dist_utils.get_rank", return_value=1)
    def test_other_ranks_evaluate_without_writing(self, rank):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            trainer._print_eval({}, Mock(), None, 0, step="les")
            self.assertEqual(trainer.evaluate.call_count, 2)
            self.assertEqual(trainer.training_history, [])
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
