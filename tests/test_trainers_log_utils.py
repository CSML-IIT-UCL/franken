import copy

import pytest
import torch

from franken.trainers.log_utils import DataSplit, HyperParameterGroup, LogCollection, LogEntry, MetricLog


@pytest.fixture
def dummy_log_dict():
    return {
        "checkpoint": {"hash": "rand_uuid", "rf_weight_id": 0},
        "timings": {"cov_coeffs": 1.0, "solve": 1.0},
        "metrics": {
            "train": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
            "validation": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
            "test": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
        },
        "hyperparameters": {
            "franken": {
                "gnn_backbone_id": "mace_mp/small",
                "interaction_block": 3,
                "kernel_type": "gaussian",
            },
            "random_features": {
                "num_random_features": 1024,
            },
            "input_scaler": {"scale_by_Z": True, "num_species": 2},
            "solver": {
                "l2_penalty": 1e-6,
                "force_weight": 0.1,
                "dtype": "torch.float64",
            },
        },
    }

def test_hpgroup_from_dict():
    dummy_group_dict = {
        "str_param": "str_value",
        "int_param": 1,
        "float_param": 1.0,
        "bool_param": True,
    }

    hpg = HyperParameterGroup.from_dict("dummy_group", dummy_group_dict)
    assert hpg.group_name == "dummy_group"
    for hp in hpg.hyperparameters:
        assert hp.name in dummy_group_dict.keys()
        assert hp.value == dummy_group_dict[hp.name]


def test_log_entry_serialize_deserialize(dummy_log_dict):
    log_entry = LogEntry.from_dict(dummy_log_dict)
    assert log_entry.to_dict() == dummy_log_dict


def test_log_entry_get_metric(dummy_log_dict):
    log_entry = LogEntry.from_dict(dummy_log_dict)
    assert log_entry.get_metric("energy_MAE", "train") == 1.0


def test_log_entry_get_invalid_metric_name(dummy_log_dict):
    log_entry = LogEntry.from_dict(dummy_log_dict)
    with pytest.raises(KeyError):
        log_entry.get_metric("invalid_metric", "train")


def test_log_entry_get_invalid_metric_split(dummy_log_dict):
    log_entry = LogEntry.from_dict(dummy_log_dict)
    with pytest.raises(KeyError):
        log_entry.get_metric("energy_MAE", "invalid_split")


class TestBestModel:
    def log_collection(self, new_metrics, dummy_log_dict) -> LogCollection:
        log_entries = []
        for metric_dict in new_metrics:
            new_log_dict = copy.copy(dummy_log_dict)
            new_log_dict["metrics"] = metric_dict
            log_entries.append(LogEntry.from_dict(new_log_dict))
        return LogCollection(logs=log_entries)
    
    def test_all_nans(self, dummy_log_dict):
        log_entries = [
            {"val": {"energy": torch.nan}},
            {"val": {"energy": torch.nan}},
        ]
        log_collection = self.log_collection(log_entries, dummy_log_dict)
        expected_best_log = MetricLog(DataSplit.VAL, "energy", torch.nan)
        best_log = log_collection.get_best_model(["energy"], split=DataSplit.VAL)
        assert len(best_log.metrics) == 1
        assert best_log.metrics[0].name == expected_best_log.name
        assert best_log.metrics[0].split == expected_best_log.split
        # assert best_log.metrics[0].value == expected_best_log.value # NaN != NaN

    def test_nans(self, dummy_log_dict):
        log_entries = [
            {"val": {"energy": torch.nan}},
            {"val": {"energy": 0.1}},
            {"val": {"energy": 12.0}},
        ]
        log_collection = self.log_collection(log_entries, dummy_log_dict)
        expected_best_log = MetricLog(DataSplit.VAL, "energy", 0.1)
        best_log = log_collection.get_best_model(["energy"], split=DataSplit.VAL)
        assert best_log.metrics == [expected_best_log]
        log_entries = [
            {"val": {"energy": 0.1}},
            {"val": {"energy": torch.nan}},
            {"val": {"energy": 12.0}},
        ]
        log_collection = self.log_collection(log_entries, dummy_log_dict)
        expected_best_log = MetricLog(DataSplit.VAL, "energy", 0.1)
        best_log = log_collection.get_best_model(["energy"], split=DataSplit.VAL)
        assert best_log.metrics == [expected_best_log]

    def test_stability(self, dummy_log_dict):
        log_entries = [
            {"val": {"energy": 1.0, "forces": 12}},
            {"val": {"energy": 1.1, "forces": 11.9}},
            {"val": {"energy": 1.2, "forces": 11.8}},
        ]
        log_collection = self.log_collection(log_entries, dummy_log_dict)
        expected_best_log = [MetricLog(DataSplit.VAL, "energy", 1.0), MetricLog(DataSplit.VAL, "forces", 12)]
        best_log = log_collection.get_best_model(["energy", "forces"], split=DataSplit.VAL)
        assert best_log.metrics == expected_best_log

    def test_normal(self, dummy_log_dict):
        log_entries = [
            {"val": {"energy": 1.0, "forces": 12}},
            {"val": {"energy": 0.9, "forces": 11.9}},
            {"val": {"energy": 1.2, "forces": 11.8}},
        ]
        log_collection = self.log_collection(log_entries, dummy_log_dict)
        expected_best_log = [MetricLog(DataSplit.VAL, "energy", 0.9), MetricLog(DataSplit.VAL, "forces", 11.9)]
        best_log = log_collection.get_best_model(["energy", "forces"], split=DataSplit.VAL)
        assert best_log.metrics == expected_best_log

    def test_missing_split(self, dummy_log_dict):
        log_entries = [
            {"val": {"energy": 1.0, "forces": 12}},
        ]
        log_collection = self.log_collection(log_entries, dummy_log_dict)
        with pytest.raises(KeyError):
            log_collection.get_best_model(["energy", "forces"], split=DataSplit.TRAIN)

    def test_missing_metric(self, dummy_log_dict):
        log_entries = [
            {"val": {"energy": 1.0, "forces": 12}},
        ]
        log_collection = self.log_collection(log_entries, dummy_log_dict)
        with pytest.raises(KeyError, match="Unknown metric='missing'"):
            log_collection.get_best_model(["missing", "forces"], split=DataSplit.VAL)
