# VulnBench

## THIS IS STILL UNDER ACTIVE DEVELOPMENT AND WILL HAVE BUGS IF YOU NOTICE ANYTHING REACH OUT OR SUBMIT A PR!

Framework for benchmarking vulnerability/defect detection models. In my travels around this area of research, there are many different datasets and models used. Some have done benchmarking on several datasets and models. However, it adds a bunch of wasted time to every research project in the space if we all have to duplicate this work.

This project glues together many different datasets of different formats and specifications into one consistent jsonl format ready for use with model training, testing and inference code. The base of that code is from `https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection` though it has been edited and expanded to use as a more flexible framework.

This framework has had some work done to automate many of the processes that have been useful in my work. However, for every use case other than just reproducibility there will probably be some required modification. For improvements that you deem useful to the wider populace feel free to submit a PR. For your own specifics, it will probably be easier just to fork this repo and make the changes you need.

# Installation

```bash
# Clone or fork this repo
git clone https://github.com/ijakenorton/VulnBench
cd VulnBench

```

## Currently this repo is easiest to setup with conda, however if you inspect the `environment.yml` file you will see the required dependencies and versions needed.
# Setup environment

# Miniconda install, only if miniconda is not already installed
# Install below from [miniconda install page](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2):
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init --all
# Source shell or reopen 
. ~/.bashrc
```

# Conda env
```bash
# This takes a few minutes to complete depending on how many of the dependencies you have cached already
conda env create -f environment.yml
conda activate ensemble
# The environment has been tested on Rocky Linux 9.2 (Blue Onyx) & Pop-os
```

# Download all datasets
```bash
cd data
./download.sh
cd ..
```

That will have setup the data directory to look like 

```bash
├── cvefixes
│   ├── README.md
│   └── cvefixes_full_dataset.jsonl
├── devign
│   ├── README.md
│   └── devign_full_dataset.jsonl
├── diversevul
│   └── diversevul_full_dataset.jsonl
...plus the rest...
```

# Usage

From here the main workflow of the framework stems from `./scripts/runner.py`. There are some utilities for exploring the current configurations. There are environment variables that can allow runner to be run from different paths, but for simplicity it is easier to be in the scripts directory from now on

```bash
cd scripts
./runner.py --list-models      # List models in config/models.json
./runner.py --list-datasets    # List datasets in config/datasets.json
# This last one is an interactive experiment picker, best to run with --dry-run to get an idea of what is happening
./runner.py --list-experiments --dry-run # List experiments in config/experiments/
```

Runner itself is designed mainly as a slurm batch submitting tool. Before running on your own slurm managed environment there is `scripts/config/hardware.json` file that will likely need to be updated. Currently it just has a string specifying the --partition you want the job to go to, however it is likely that additional information will be needed to be loaded from here in the future. However as I only have access to my local University cluster I am unsure what else would be needed here.

After having a look around in the config folder for the current configurations and experiment schema, run the runner with --dry-run on the experiment you want to make sure it will be submitting the correct batch of jobs.

# Logging

All logs go to the `./logs` or `LOGS_DIR` folder. The names of each run are built in a consistent format based on the hyperparameters, read  the `_generate_experiment_canonical_name` function in `scripts/runner.py` for more details though an example name would be:

```
    Format: {dataset}_{seed}_{suffix}_{hyperparam_changes}
    Example: devign_seed123456_splits_pos2.0_lr5e-5
```

If you want to check why an error occured there is a script `./scripts/aggregate_errors.py` that will scan for logs from the last batch of runs. I'm sure this will have some cases that it misses, but in my person experience it is a great starting point for tracing bugs.

## History

Each submitted batch is added to a `history.json` file, this can be shown through `./scripts/view_run_history.py`

## Weights and Biases

All runs will be submitted to [Weights and Biases](https://wandb.ai/site) by default, though this can be configured. 

# Results Aggregation and Analysis

After model training has finished, run `./scripts/aggregate_results_threshold.py` to see the results of all current runs within `./models`. This will output a report showing the experiments that succeeded, and failed as well as the results in this sort of format:

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

```

# Aggregate all results with threshold optimization

```bash
# The below also emit <output_name>.json which can be used with other scripts
python aggregate_results_threshold.py # Will default to output file of artifacts/results_summary.csv
python aggregate_results_threshold.py --results_dir ../models --output results_summary.csv
```

The JSON file output by ` aggregate_results_threshold` can be used with `./scripts/find_best_model_examples.py` which is a script used to find interesting trends with the model predictions, code examples which only 1 model got right for example. This is here as an example of what the results output format can be used for.

# Brittle

Bare in mind that some of the systems within this repo may be brittle as I am only a solo dev that has made this in conjuction with doing masters, so outside of the main use cases expect to have to do a bit of work.


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

## Adding New Models or Datasets

1. **Add a new model**: Edit `config/models.json` and add your model configuration
2. **Add a new dataset**: Edit `config/datasets.json` with dataset path and metadata
3. **Create experiment**: Create a new experiment config in `config/experiments/`
4. **Run**: `python runner.py config/experiments/your_experiment.json`

# Hardware Requirements

Currently I run all the models on H100s with 64gb of RAM. I believe most of the datasets will not need such a heavy duty setup. Draper and DiverseVul are very large datasets and will most likely be more difficult to run on smaller GPUs. Modifying the batch sizes in the model configs may help this though.

The framework automatically handles smaller vs. larger datasets differently - smaller datasets get 20GB GPU jobs while the big ones (DiverseVul, Draper) get 50GB H100 jobs.
