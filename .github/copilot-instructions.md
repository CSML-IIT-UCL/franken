# Franken: AI Coding Agent Instructions

**Franken** is a PyTorch-based ML library for fine-tuning atomistic foundation models with random-feature approximations.

## Architecture Overview

### Core Components

- **Backbones** (`franken/backbones/`): Wrappers for foundation models (MACE, SchNet, SevenNet) with dynamic registry (`registry.json`) - models are loaded lazily and cached
- **Random Features** (`franken/rf/`): Core inference engine - `FrankenPotential` combines backbone descriptors with learned RF weights via `RFTrainer`
- **Trainers** (`franken/trainers/`): `BaseTrainer` + `RFTrainer` implement covariance matrix computation (via JAC) and least-squares solve - `LogCollection` tracks metrics/hyperparameters
- **Metrics** (`franken/metrics/`): Singleton registry for computing MAE/RMSE on energy/forces; includes Pareto frontier selection
- **Datasets** (`franken/datasets/`): Registry pattern for multi-split datasets (water, TM23, PtH2O); auto-download on first access
- **Autotune** (`franken/autotune/`): CLI-driven hyperparameter search using `AutotuneConfig` dataclasses via OmegaConf

### Data Flow

1. User provides train/val XYZ files or selects registered dataset
2. Backbone computes atomic descriptors (features) via forward pass
3. `RFTrainer.fit()` computes covariance matrix (`cov_coeffs`) and solves least-squares (`solve`)
4. Metrics computed on validation split; best model selected by Pareto norm (`get_best_model()`)
5. Trained model saved as `FrankenCalculator` for ASE/LAMMPS inference

## Developer Workflows

### Running Tests
```bash
pytest tests/
pytest tests/test_trainer.py -v  # Specific test file
```
- Tests auto-download MACE checkpoints to `.franken/` cache
- Set `FRANKEN_CACHE_DIR` env var to override cache location
- Parameterized tests use `@pytest.mark.parametrize` for multi-device runs

### Code Quality
```bash
black franken/ tests/  # Format (enforce in pre-commit)
ruff check --fix franken/ tests/  # Lint
```
- **Black** and **Ruff** are mandatory in CI; run before committing
- Linting runs on every push

### Building Documentation
```bash
cd docs && make html
```
- Sphinx with MyST parser; hosted on ReadTheDocs
- Notebooks in `docs/notebooks/` are auto-included

## Project-Specific Patterns

### Registry Pattern (Multi-component)

**Metrics** (`franken/metrics/registry.py`):
```python
# Definition in franken/metrics/functions.py
class EnergyMAE(BaseMetric): ...
registry.register("energy_MAE", EnergyMAE)

# Usage
metric = franken.metrics.init_metric("energy_MAE", device)
```

**Datasets** (`franken/datasets/registry.py`):
```python
# Decorator registration
@DATASET_REGISTRY.register("water")
class WaterRegisteredDataset(BaseRegisteredDataset):
    relative_paths = {...}
    @classmethod
    def get_path(name, split, base_path): ...
```

**Backbones** (`franken/backbones/utils.py`): JSON-based registry (`registry.json`) with remote/local URLs for lazy downloading

### Config System

Uses **frozen dataclasses** with OmegaConf for CLI parsing:
- `AutotuneConfig` → `MaceBackboneConfig`, `GaussianRFConfig`, `SolverConfig`, etc.
- Supports hyperparameter search ranges: `HPSearchConfig(start=1, stop=10, num=5)`
- CLI docstrings auto-parsed by `docstring_parser` for argument help

### Distributed Training

`franken/utils/distributed.py` provides `all_gather_object()`, `all_sum()` for multi-GPU:
- `LogCollection.gather_from_ranks()` aggregates logs across ranks
- `BaseMetric.compute()` reduces metrics via `dist_utils.all_sum()`

### Logging & Checkpointing

- **JSON-based logging**: Each training iteration → `LogEntry` → JSON file (`log.json`)
- `LogCollection.save_json()` appends batch to file; `from_json()` loads
- Checkpoint tracking via `checkpoint_hash` + `checkpoint_rf_weight_id`

## Critical Integration Points

### Backbone Loading
- `franken/backbones/utils.py::get_checkpoint_path()`: Validates backbone name, downloads if needed
- Supports MACE/SchNet/SevenNet via `franken/backbones/wrappers/` adapters

### Dataset Access
- `DatasetRegistry.get_path(name, split, base_path)` returns ASE-readable file path
- Auto-downloads via dataset class's `download()` classmethod
- Check `is_valid_split()` before querying splits

### Metrics & Pareto Selection
- `is_pareto_efficient(costs)`: NumPy-based Pareto frontier computation
- `get_best_model()` minimizes p-norm over Pareto-efficient models
- Metrics returned as lists (one value per log entry) for vectorized operations

## Testing & Validation

- **Fixtures** in `conftest.py`: `random_seed` (auto-used), `DEVICES` (CPU + CUDA variants)
- **Mocking**: Mock registries in `test_backbones_utils.py` for isolation
- **Skip markers**: `@SKIP_NO_CUDA` for GPU-only tests
- Coverage: `pytest --cov=franken --cov-report=html`

## Common Pitfalls

1. **Registry not imported**: Ensure submodules (e.g., `from .water import water_dataset`) are in `__init__.py` before registry access
2. **Cache directory**: Tests use `.franken/` - clear if corrupted
3. **DataSplit enum**: Use `DataSplit.TRAIN`/`DataSplit.VAL`/`DataSplit.TEST` (case-sensitive in `from_dict()`)
4. **Metric values**: May be `NaN` if split not computed; `get_best_model()` converts to `np.inf` for filtering
5. **Backbone dtype mismatch**: RF model dtype must match backbone dtype; check `BaseTrainer.__init__()`

## Key Files Reference

- `pyproject.toml`: Dependencies, CLI entry points, optional extras (cuda, mace, fairchem, sevenn, docs)
- `tests/conftest.py`: Global pytest setup, device fixtures, random seed initialization
- `franken/config.py`: Central config system with `all_fields()` and `asdict_with_classvar()`
- `docs/topics/model_registry.md`: Backbone registry documentation and usage examples
