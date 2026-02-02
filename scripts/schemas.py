"""Configuration schemas and validation for vulnerability detection experiments."""

from dataclasses import dataclass
from typing import List, Literal, Optional
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str
    tokenizer_name: str
    model_type: str


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    size: Literal["small", "big"]
    time_hours: int
    gpu: str


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    models: List[str]
    datasets: List[str]
    seeds: List[int]
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
    use_wandb: bool = True
    wandb_project: str = "vulnerability-benchmark"
    use_original_splits: bool = False
    # Cross-dataset testing: specify source model directory pattern
    # e.g., "primevul_seed{seed}" to test PrimeVul-trained model on other datasets
    source_model_dir: Optional[str] = None
    # Loss function configuration
    loss_type: Literal["bce", "cb_focal"] = "bce"
    cb_beta: float = 0.9999
    focal_gamma: float = 2.0
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
        self._dataset_groups = None

    def load_models(self) -> dict[str, ModelConfig]:
        """Load model configurations."""
        if self._models is None:
            models_file = self.config_dir / "models.json"
            try:
                with open(models_file) as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Error parsing {models_file}: {e.msg}",
                    e.doc, e.pos
                ) from e
            self._models = {
                name: ModelConfig(**config)
                for name, config in data["models"].items()
            }
        return self._models

    def load_datasets(self) -> dict[str, DatasetConfig]:
        """Load dataset configurations."""
        if self._datasets is None:
            datasets_file = self.config_dir / "datasets.json"
            try:
                with open(datasets_file) as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Error parsing {datasets_file}: {e.msg}",
                    e.doc, e.pos
                ) from e
            self._datasets = {
                name: DatasetConfig(**config)
                for name, config in data["datasets"].items()
            }

            # Auto-generate dataset groups from "size" field if not provided
            if "dataset_groups" in data:
                self._dataset_groups = data["dataset_groups"]
            else:
                self._dataset_groups = self._generate_dataset_groups()
        return self._datasets

    def _generate_dataset_groups(self) -> dict[str, List[str]]:
        """Auto-generate dataset groups based on the 'size' field."""
        groups = {
            "small": [],
            "big": [],
            "all": []
        }

        for name, config in self._datasets.items():
            groups["all"].append(name)
            if config.size == "small":
                groups["small"].append(name)
            elif config.size == "big":
                groups["big"].append(name)

        # Sort for consistency
        for group in groups.values():
            group.sort()

        return groups

    def get_dataset_group(self, group_name: str) -> List[str]:
        """Get a list of datasets by group name."""
        if self._dataset_groups is None:
            self.load_datasets()
        return self._dataset_groups.get(group_name, [])

    def load_experiment(self, experiment_file: Path) -> ExperimentConfig:
        """Load an experiment configuration."""
        try:
            with open(experiment_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Error parsing {experiment_file}: {e.msg}",
                e.doc, e.pos
            ) from e

        # Expand dataset groups
        datasets = []
        for ds in data.get("datasets", []):
            if ds.startswith("group:"):
                group_name = ds.replace("group:", "")
                datasets.extend(self.get_dataset_group(group_name))
            else:
                datasets.append(ds)

        data["datasets"] = datasets
        # Remove metadata fields that aren't part of the dataclass
        data.pop("_metadata", None)
        return ExperimentConfig(**data)

    def validate_experiment(self, experiment: ExperimentConfig) -> List[str]:
        """Validate an experiment configuration and return any errors."""
        errors = []

        models = self.load_models()
        datasets = self.load_datasets()

        # Check all models exist
        for model in experiment.models:
            if model not in models:
                errors.append(f"Unknown model: {model}")

        # Check all datasets exist
        for dataset in experiment.datasets:
            if dataset not in datasets:
                errors.append(f"Unknown dataset: {dataset}")

        # Validate seeds
        if not experiment.seeds:
            errors.append("At least one seed is required")

        return errors
