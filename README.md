# VulnBench

Framework for benchmarking vulnerability/defect detection models. In my travels around this area of research, there are many different datasets and models used. Some have done benchmarking on several datasets and models. However, it adds a bunch of wasted time to every research project in the space if we all have to duplicate this work.

This project glues together 9 different datasets of different formats and specifications into one consistent jsonl format ready for use with model training, testing and inference code. The base of that code is from `https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection` though it has been edited and expanded to use as a more flexible framework.

This is designed to be forked and extended by the user. It is a base to work from. If you need something, generally you will need to edit the code itself. The hope is that as it is small and mostly contained this will be easy.

## Datasets

All datasets are now available on Hugging Face and will be automatically downloaded:

- **BigVul (dedup)** - Deduplicated version of BigVul dataset from Croft et al. 2023
- **CVEFixes** - Real-world vulnerability fixes from CVE-linked GitHub commits
- **Devign (dedup)** - Deduplicated version of Devign (Qemu and ffmpeg functions with errors)
- **DiverseVul** - Real-world vulnerability data from git commits, C/C++
- **Draper** - Large aggregation of C/C++ real-world code from GitHub scraping
- **ICVul** - Collection of vulnerability contributing commits from CVEs linked to GitHub commits
- **Juliet (dedup)** - Deduplicated version of NIST's synthetic test suite for C/C++ vulnerabilities
- **MVDSC Mixed** - Mixed formats, some in AST/Graph form, includes Juliet samples
- **PrimeVul** - Vulnerability detection dataset
- **Reveal** - Cleaned real-world dataset from Chrome and Debian, JSON format
- **VulDeepecker** - From SARD dataset, C/C++, mix of synthetic and real vulnerabilities

The datasets range from highly synthetic (Juliet) to completely real-world (CVEFixes, DiverseVul) with various complexity levels. The deduplicated datasets (dedup-bigvul, dedup-devign, dedup-juliet) are based on the data quality work by Croft et al. (2023) which removes duplicates and improves dataset quality.

## Models

Currently supports these pre-trained models:
- **CodeBERT** - Microsoft's code-understanding model (BERT-based encoder)
- **CodeT5** - Salesforce's encoder-only variant (T5EncoderModel)
- **CodeT5-Full** - Salesforce's full encoder-decoder model (T5ForConditionalGeneration)
- **GraphCodeBERT** - Microsoft's graph-based code model with data flow
- **LineVul** - CodeBERT-based model specifically designed for line-level vulnerability detection
- **NatGen** - Natural language to code generation model

# Usage

```bash
# Clone or fork this repo
git clone https://github.com/ijakenorton/defect_detection_benchmark
cd defect_detection_benchmark

# Setup environment
conda env create -f environment.yml
conda activate ensemble
# The environment has been tested on Rocky Linux 9.2 (Blue Onyx) & Pop-os

# Download all datasets (now from Hugging Face!)
cd data
./download.sh
cd ..

# Train models using the Python-based configuration system
cd scripts

# Train all models on all datasets with multiple seeds
python runner.py config/experiments/train_all.json

# Only run missing experiments (automatically detects what's already completed)
python runner.py config/experiments/train_all.json --fix-missing

# Preview what would be run without executing
python runner.py config/experiments/train_all.json --dry-run

# Run locally without SLURM/sbatch
python runner.py config/experiments/train_all.json --no-sbatch

# Find what experiments are missing
python find_missing_experiments.py --config config/experiments/train_all.json

# Aggregate results across all experiments
python aggregate_results_threshold.py --results_dir ../models --output results_summary.csv
```

## What's Actually Implemented

This framework is now pretty mature and includes:

- **Automated dataset downloading** from Hugging Face
- **Multi-seed training** for robust statistical results
- **JSON-based configuration system** - centralized configs for models, datasets, and experiments
- **Smart experiment tracking** - automatically detects completed experiments and runs only what's missing
- **Batch job support** - works with SLURM for cluster training
- **Results aggregation** - automatically combines results across seeds and experiments
- **Flexible training scripts** - train individual models/datasets or run the full suite
- **Legacy and new directory support** - handles both old bash-based and new Python-based experiment outputs

## Project Structure

```
├── data/                    # Dataset handling and transformation scripts
│   ├── download.sh         # Downloads all datasets from Hugging Face
│   └── */                  # Individual dataset processing scripts
├── scripts/                # Training and evaluation scripts
│   ├── runner.py          # Main experiment runner (replaces bash scripts)
│   ├── schemas.py         # Configuration schema definitions and validation
│   ├── find_missing_experiments.py  # Detect which experiments still need to be run
│   ├── aggregate_results_threshold.py  # Combine results across seeds
│   ├── config/            # Configuration files
│   │   ├── models.json    # Model definitions (CodeBERT, CodeT5, etc.)
│   │   ├── datasets.json  # Dataset configurations
│   │   └── experiments/   # Experiment configurations
│   │       └── train_all.json  # Train all models on all datasets
│   └── *.sh               # Legacy bash scripts (being phased out)
├── models/                 # Pre-trained models and experiment outputs
│   ├── codebert/          # CodeBERT experiment results
│   ├── codet5/            # CodeT5 (encoder-only) results
│   ├── codet5-full/       # CodeT5 (full encoder-decoder) results
│   └── */                 # Other model results
└── Defect-detection/      # Core training/inference code
```

## Documentation

- **[Runner.py Guide](scripts/RUNNER_README.md)** - Comprehensive guide to the experiment runner
- **[Analysis Pipeline](scripts/ANALYSIS_README.md)** - Data analysis and visualization tools
- **[GNN Integration](scripts/GNN_INTEGRATION_README.md)** - Graph Neural Network experiments

## Advanced Usage

### Configuration System

The framework uses a centralized JSON-based configuration system:

- **`config/models.json`** - Define models with their HuggingFace paths and types
- **`config/datasets.json`** - Dataset configurations with paths and metadata
- **`config/experiments/*.json`** - Experiment definitions (model/dataset/seed combinations)

Datasets are automatically grouped by size (small/big) based on their `size` field, eliminating manual group management.

For detailed configuration options and advanced features (cross-dataset testing, original splits, hyperparameter tracking), see [scripts/RUNNER_README.md](scripts/RUNNER_README.md).

### Custom Experiments

```bash
# Create a custom experiment config
cat > config/experiments/my_experiment.json <<EOF
{
  "models": ["codebert", "codet5"],
  "datasets": ["juliet", "devign"],
  "seeds": [123456, 789012],
  "pos_weight": 1.0,
  "epoch": 5,
  "out_suffix": "splits",
  "mode": "train"
}
EOF

# Run your custom experiment
python runner.py config/experiments/my_experiment.json

# Or just run what's missing
python runner.py config/experiments/my_experiment.json --fix-missing
```

### Working with Results

```bash
# Check experiment completion status
python find_missing_experiments.py --config config/experiments/train_all.json

# Aggregate all results with threshold optimization
python aggregate_results_threshold.py --results_dir ../models --output results_summary.csv

# Test trained models (legacy bash script)
./test_all_models_datasets_std.sh
```

### Adding New Models or Datasets

1. **Add a new model**: Edit `config/models.json` and add your model configuration
2. **Add a new dataset**: Edit `config/datasets.json` with dataset path and metadata
3. **Create experiment**: Create a new experiment config in `config/experiments/`
4. **Run**: `python runner.py config/experiments/your_experiment.json`

# Hardware Requirements

Currently I run all the models on H100s with 64gb of RAM. I believe most of the datasets will not need such a heavy duty setup. Draper and DiverseVul are very large datasets and will most likely be more difficult to run on smaller GPUs. Modifying the batch sizes in the model configs may help this though.

The framework automatically handles smaller vs. larger datasets differently - smaller datasets get 20GB GPU jobs while the big ones (DiverseVul, Draper) get 50GB H100 jobs.

I intend to do some testing on what are the minimal specs required for each of the default training options.

# Current Results

## Summary

---

```
             Model             Dataset  Pos Weight  Seeds            F1      Accuracy     Precision        Recall
     codebert-base     cvefixes_splits         1.0      3 0.598 ± 0.006 0.458 ± 0.012 0.432 ± 0.005 0.972 ± 0.022 
     codebert-base       devign_splits         1.0      3 0.660 ± 0.007 0.582 ± 0.013 0.524 ± 0.009 0.892 ± 0.011 
     codebert-base       draper_splits         1.0      3 0.377 ± 0.326 0.936 ± 0.001 0.561 ± 0.092 0.424 ± 0.367 
     codebert-base        icvul_splits         1.0      3 0.585 ± 0.002 0.436 ± 0.004 0.418 ± 0.002 0.972 ± 0.003 
     codebert-base       juliet_splits         1.0      3 0.896 ± 0.002 0.940 ± 0.003 0.849 ± 0.023 0.950 ± 0.025 
     codebert-base       reveal_splits         1.0      3 0.468 ± 0.024 0.881 ± 0.013 0.422 ± 0.035 0.540 ± 0.104 
     codebert-base vuldeepecker_splits         1.0      3 0.958 ± 0.004 0.976 ± 0.002 0.969 ± 0.009 0.947 ± 0.011 
       codet5-base     cvefixes_splits         1.0      3 0.592 ± 0.002 0.440 ± 0.013 0.424 ± 0.005 0.980 ± 0.021 
       codet5-base       devign_splits         1.0      3 0.671 ± 0.010 0.600 ± 0.035 0.539 ± 0.026 0.892 ± 0.054 
       codet5-base   diversevul_splits         1.0      3 0.302 ± 0.030 0.901 ± 0.019 0.260 ± 0.040 0.371 ± 0.061 
       codet5-base       draper_splits         1.0      3 0.609 ± 0.007 0.947 ± 0.000 0.583 ± 0.001 0.639 ± 0.015 
       codet5-base        icvul_splits         1.0      3 0.586 ± 0.002 0.434 ± 0.011 0.418 ± 0.004 0.979 ± 0.013 
       codet5-base       juliet_splits         1.0      3 0.900 ± 0.003 0.941 ± 0.002 0.839 ± 0.005 0.969 ± 0.000 
       codet5-base       reveal_splits         1.0      3 0.485 ± 0.017 0.879 ± 0.013 0.420 ± 0.031 0.583 ± 0.080 
       codet5-base vuldeepecker_splits         1.0      3 0.959 ± 0.001 0.977 ± 0.001 0.976 ± 0.010 0.942 ± 0.008 
graphcodebert-base     cvefixes_splits         1.0      3 0.603 ± 0.006 0.477 ± 0.026 0.440 ± 0.011 0.959 ± 0.024 
graphcodebert-base       devign_splits         1.0      3 0.667 ± 0.006 0.583 ± 0.006 0.524 ± 0.004 0.916 ± 0.013 
graphcodebert-base       draper_splits         1.0      3 0.562 ± 0.015 0.938 ± 0.003 0.515 ± 0.021 0.618 ± 0.008 
graphcodebert-base        icvul_splits         1.0      3 0.586 ± 0.005 0.437 ± 0.019 0.419 ± 0.008 0.974 ± 0.016 
graphcodebert-base       juliet_splits         1.0      3 0.896 ± 0.004 0.939 ± 0.004 0.829 ± 0.019 0.976 ± 0.019 
graphcodebert-base       reveal_splits         1.0      3 0.486 ± 0.007 0.886 ± 0.011 0.437 ± 0.029 0.555 ± 0.069 
graphcodebert-base vuldeepecker_splits         1.0      3 0.958 ± 0.001 0.976 ± 0.001 0.974 ± 0.004 0.944 ± 0.002 
            natgen     cvefixes_splits         1.0      3 0.594 ± 0.009 0.447 ± 0.020 0.427 ± 0.009 0.976 ± 0.002 
            natgen       devign_splits         1.0      3 0.669 ± 0.006 0.598 ± 0.021 0.536 ± 0.017 0.892 ± 0.037 
            natgen   diversevul_splits         1.0      3 0.307 ± 0.025 0.891 ± 0.013 0.243 ± 0.023 0.422 ± 0.061 
            natgen       draper_splits         1.0      3 0.599 ± 0.006 0.946 ± 0.003 0.577 ± 0.026 0.623 ± 0.018 
            natgen        icvul_splits         1.0      3 0.586 ± 0.003 0.430 ± 0.007 0.416 ± 0.003 0.988 ± 0.005 
            natgen       juliet_splits         1.0      3 0.900 ± 0.003 0.942 ± 0.002 0.841 ± 0.004 0.968 ± 0.003 
            natgen       reveal_splits         1.0      3 0.481 ± 0.044 0.880 ± 0.025 0.427 ± 0.074 0.558 ± 0.025 
            natgen vuldeepecker_splits         1.0      3 0.955 ± 0.002 0.975 ± 0.001 0.974 ± 0.002 0.937 ± 0.005 
```

## Threshold Optimization Improvement Analysis

---

```
             Model             Dataset  Seeds Default F1 (0.5) Optimal Threshold F1 Improvement Relative Gain
     codebert-base     cvefixes_splits      3    0.137 ± 0.115     0.167 ± 0.046 +0.461 ± 0.109       +337.7%
     codebert-base       devign_splits      3    0.514 ± 0.024     0.167 ± 0.061 +0.146 ± 0.029        +28.4%
     codebert-base       draper_splits      3    0.333 ± 0.288     0.133 ± 0.031 +0.044 ± 0.038        +13.2%
     codebert-base        icvul_splits      3    0.051 ± 0.079     0.273 ± 0.090 +0.534 ± 0.079      +1051.4%
     codebert-base       juliet_splits      3    0.894 ± 0.003     0.453 ± 0.031 +0.003 ± 0.004         +0.3%
     codebert-base       reveal_splits      3    0.393 ± 0.052     0.133 ± 0.031 +0.074 ± 0.042        +18.8%
     codebert-base vuldeepecker_splits      3    0.955 ± 0.003     0.613 ± 0.341 +0.002 ± 0.000         +0.3%
       codet5-base     cvefixes_splits      3    0.106 ± 0.090     0.293 ± 0.042 +0.486 ± 0.089       +461.0%
       codet5-base       devign_splits      3    0.614 ± 0.010     0.260 ± 0.122 +0.056 ± 0.001         +9.2%
       codet5-base   diversevul_splits      3    0.069 ± 0.032     0.113 ± 0.023 +0.234 ± 0.042       +340.5%
       codet5-base       draper_splits      3    0.567 ± 0.017     0.113 ± 0.023 +0.043 ± 0.018         +7.5%
       codet5-base        icvul_splits      3    0.060 ± 0.045     0.187 ± 0.046 +0.526 ± 0.043       +874.5%
       codet5-base       juliet_splits      3    0.896 ± 0.008     0.420 ± 0.060 +0.004 ± 0.005         +0.4%
       codet5-base       reveal_splits      3    0.373 ± 0.058     0.147 ± 0.050 +0.113 ± 0.061        +30.2%
       codet5-base vuldeepecker_splits      3    0.957 ± 0.001     0.593 ± 0.253 +0.002 ± 0.001         +0.2%
graphcodebert-base     cvefixes_splits      3    0.166 ± 0.089     0.200 ± 0.072 +0.438 ± 0.089       +264.2%
graphcodebert-base       devign_splits      3    0.523 ± 0.059     0.127 ± 0.023 +0.144 ± 0.054        +27.6%
graphcodebert-base       draper_splits      3    0.508 ± 0.004     0.160 ± 0.035 +0.054 ± 0.015        +10.6%
graphcodebert-base        icvul_splits      3    0.059 ± 0.071     0.253 ± 0.061 +0.527 ± 0.069       +898.4%
graphcodebert-base       juliet_splits      3    0.893 ± 0.008     0.340 ± 0.209 +0.004 ± 0.005         +0.4%
graphcodebert-base       reveal_splits      3    0.396 ± 0.047     0.180 ± 0.053 +0.090 ± 0.043        +22.8%
graphcodebert-base vuldeepecker_splits      3    0.957 ± 0.001     0.573 ± 0.012 +0.001 ± 0.001         +0.1%
            natgen     cvefixes_splits      3    0.117 ± 0.147     0.207 ± 0.083 +0.477 ± 0.138       +409.4%
            natgen       devign_splits      3    0.609 ± 0.004     0.160 ± 0.035 +0.061 ± 0.008        +10.0%
            natgen   diversevul_splits      3    0.084 ± 0.030     0.120 ± 0.035 +0.223 ± 0.031       +264.5%
            natgen       draper_splits      3    0.568 ± 0.014     0.153 ± 0.092 +0.031 ± 0.007         +5.4%
            natgen        icvul_splits      3    0.044 ± 0.050     0.180 ± 0.040 +0.542 ± 0.047      +1232.0%
            natgen       juliet_splits      3    0.893 ± 0.004     0.380 ± 0.035 +0.006 ± 0.001         +0.7%
            natgen       reveal_splits      3    0.347 ± 0.048     0.127 ± 0.031 +0.133 ± 0.005        +38.4%
            natgen vuldeepecker_splits      3    0.954 ± 0.001     0.520 ± 0.209 +0.001 ± 0.001         +0.1%
```

## Best Models Per Dataset

---

```
cvefixes_splits: graphcodebert-base F1=0.603±0.006 (3 seeds)
devign_splits  : codet5-base  F1=0.671±0.010 (3 seeds)
draper_splits  : codet5-base  F1=0.609±0.007 (3 seeds)
icvul_splits   : natgen       F1=0.586±0.003 (3 seeds)
juliet_splits  : natgen       F1=0.900±0.003 (3 seeds)
reveal_splits  : graphcodebert-base F1=0.486±0.007 (3 seeds)
vuldeepecker_splits: codet5-base  F1=0.959±0.001 (3 seeds)
diversevul_splits: natgen       F1=0.307±0.025 (3 seeds)
```

## Improvement Summary Statistics

---

```
Average F1 improvement: +0.182
Median F1 improvement:  +0.082
Best improvement:       +0.542
Worst improvement:      +0.001
Positive improvements:  30/30 (100.0%)
```
