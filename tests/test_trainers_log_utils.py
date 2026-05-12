import pytest

from franken.trainers.log_utils import HyperParameterGroup, LogEntry


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


# class TestBestModel:
#     def test_all_nans(self):
#         log_entries = [
#             {"metrics": {"val": {"energy": torch.nan}}},
#             {"metrics": {"val": {"energy": torch.nan}}},
#         ]
#         expected_best_log = log_entries[0]
#         best_log = get_best_model(log_entries, ["energy"], split="val")
#         assert best_log == expected_best_log

#     def test_nans(self):
#         log_entries = [
#             {"metrics": {"val": {"energy": torch.nan}}},
#             {"metrics": {"val": {"energy": 0.1}}},
#             {"metrics": {"val": {"energy": 12.0}}},
#         ]
#         expected_best_log = log_entries[1]
#         best_log = get_best_model(log_entries, ["energy"], split="val")
#         assert best_log == expected_best_log
#         log_entries = [
#             {"metrics": {"val": {"energy": 0.1}}},
#             {"metrics": {"val": {"energy": torch.nan}}},
#             {"metrics": {"val": {"energy": 12.0}}},
#         ]
#         expected_best_log = log_entries[0]
#         best_log = get_best_model(log_entries, ["energy"], split="val")
#         assert best_log == expected_best_log

#     def test_stability(self):
#         log_entries = [
#             {"metrics": {"val": {"energy": 1.0, "forces": 12}}},
#             {"metrics": {"val": {"energy": 1.1, "forces": 11.9}}},
#             {"metrics": {"val": {"energy": 1.2, "forces": 11.8}}},
#         ]
#         expected_best_log = log_entries[0]
#         best_log = get_best_model(log_entries, ["energy", "forces"], split="val")
#         assert best_log == expected_best_log

#     def test_normal(self):
#         log_entries = [
#             {"metrics": {"val": {"energy": 1.0, "forces": 12}}},
#             {"metrics": {"val": {"energy": 0.9, "forces": 11.9}}},
#             {"metrics": {"val": {"energy": 1.2, "forces": 11.8}}},
#         ]
#         expected_best_log = log_entries[1]
#         best_log = get_best_model(log_entries, ["energy", "forces"], split="val")
#         assert best_log == expected_best_log

#     def test_missing_split(self):
#         log_entries = [
#             {"metrics": {"val": {"energy": 1.0, "forces": 12}}},
#         ]
#         with pytest.raises(KeyError):
#             get_best_model(log_entries, ["energy", "forces"], split="train")

#     def test_missing_metric(self):
#         log_entries = [
#             {"metrics": {"val": {"energy": 1.0, "forces": 12}}},
#         ]
#         with pytest.raises(KeyError):
#             get_best_model(log_entries, ["missing", "forces"], split="val")
