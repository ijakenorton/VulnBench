"""Configuration schemas and validation for vulnerability detection experiments."""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
import json
from pathlib import Path
import sys


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str
    tokenizer_name: str
    model_type: str


@dataclass
class HardwareConfig:
    """Hardware configuration."""

    # TODO May need added fields like
    # nodes or cpu or mem specs

    # mem: str
    partition: str


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    # TODO unsure if dataset should have gpu field still as maybe finer grained control is good

    name: str
    size: Literal["small", "big"]
    time_hours: int


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    models: List[str]
    datasets: List[str]
    hardware: dict[str, HardwareConfig]

    seeds: List[int]
    # Model hyperparameters
    pos_weight: float = 1.0
    epoch: int = 5
    out_suffix: str = ""
    mode: Literal["train", "test"] = "train"
    block_size: int = 400
    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    dropout_probability: float = 0.2

    # Loss function configuration
    loss_type: Literal["bce", "cb_focal"] = "bce"
    cb_beta: float = 0.9999
    focal_gamma: float = 2.0

    # Whether to log to wandb
    use_wandb: bool = True
    wandb_project: str = "vulnerability-benchmark"

    use_original_splits: bool = False
    # Cross-dataset testing: specify source model directory pattern
    # e.g., "primevul_seed{seed}" to test PrimeVul-trained model on other datasets
    source_model_dir: Optional[str] = None

    # Threshold optimization (inference time - separate from pos_weight which affects training)
    threshold_method: Literal["grid_search", "ghost", "both"] = "both"
    threshold_metric: Literal["f1", "precision", "kappa", "mcc"] = "f1"
    min_recall: float = 0.5
    threshold_precision_weight: float = 2.0
    ghost_n_subsets: int = 100
    ghost_subset_size: float = 0.8

    def __getitem__(self, item):
        return getattr(self, item)


class ConfigLoader:
    """Loads and validates configuration files."""

    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self._models = None
        self._datasets = None
        self._hardware = None
        self._dataset_groups = None

    def log_json_parse_error(self, config_file, e):
        print(
            f"Error parsing {config_file}: {e.msg}",
            e.doc,
            e.pos,
            file=sys.stderr,
        )
        sys.exit(1)

    def load_models(self) -> dict[str, ModelConfig]:
        """Load model configurations."""
        data = None
        if self._models is None:
            models_file = self.config_dir / "models.json"
            try:
                with open(models_file) as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                self.log_json_parse_error(models_file, e)

            if data is None:
                print("failed to load models file exiting...", file=sys.stderr)
                sys.exit(1)

            self._models = {
                name: ModelConfig(**config) for name, config in data["models"].items()
            }
        return self._models

    def load_hardware(self) -> dict[str, HardwareConfig]:
        """Load model configurations."""
        data = None
        hardware_file = self.config_dir / "hardware.json"
        if self._hardware is None:
            try:
                with open(hardware_file) as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                self.log_json_parse_error(hardware_file, e)

            if data is None:
                print("failed to load hardware file exiting...", file=sys.stderr)
                sys.exit(1)
            self._hardware = {
                name: HardwareConfig(**config) for name, config in data.items()
            }

            if self._hardware is None:
                print(
                    "Hardware file is non, check",
                    hardware_file,
                    "is not empty",
                    file=sys.stderr,
                )
                sys.exit(1)
        return self._hardware

    def load_datasets(self) -> dict[str, DatasetConfig]:
        """Load dataset configurations."""
        data = None
        if self._datasets is None:
            datasets_file = self.config_dir / "datasets.json"
            try:
                with open(datasets_file) as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                self.log_json_parse_error(datasets_file, e)
            if data is None:
                print("failed to load dataset file exiting...", file=sys.stderr)
                sys.exit(1)
            self._datasets = {
                name: DatasetConfig(**config)
                for name, config in data["datasets"].items()
            }

        return self._datasets

    def load_experiment(self, experiment_file: Path) -> ExperimentConfig:
        """Load an experiment configuration."""
        data = None
        try:
            with open(experiment_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.log_json_parse_error(experiment_file, e)

        if data is None:
            print("failed to load experiment file exiting...", file=sys.stderr)
            sys.exit(1)
        # Expand dataset groups
        datasets = []
        for ds in data.get("datasets", []):
            datasets.append(ds)
        data["datasets"] = datasets

        models = []
        for ds in data.get("models", []):
            models.append(ds)
        data["models"] = models
        data.pop("_metadata", None)

        data["hardware"] = self.load_hardware()
        return ExperimentConfig(**data)

    def load_config_files(
        self,
    ) -> Tuple[
        dict[str, ModelConfig], dict[str, DatasetConfig], dict[str, HardwareConfig]
    ]:
        """Load models and datasets config files."""
        models = self.load_models()
        datasets = self.load_datasets()
        hardware = self.load_hardware()

        return models, datasets, hardware

    def validate_experiment(self, experiment: ExperimentConfig) -> List[str]:
        """Validate an experiment configuration and return any errors."""
        errors = []
        models, datasets, hardware = self.load_config_files()

        # Check all models exist
        for model in experiment.models:
            if model not in models:
                errors.append(f"Unknown model: {model}")

        # Check all datasets exist
        for dataset in experiment.datasets:
            if dataset not in datasets:
                errors.append(f"Unknown dataset: {dataset}")

        # Check all hardware gpus exist
        for dataset in datasets.values():
            if dataset.size not in hardware:
                errors.append(f"Unknown gpu name: {dataset.size}")

        # Validate seeds
        if not experiment.seeds:
            errors.append("At least one seed is required")

        return errors
