#!/bin/bash
#SBATCH --job-name=spurious_corr
#SBATCH --output=../logs/spurious_correlations_%j.out
#SBATCH --error=../logs/spurious_correlations_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=batch

# Spurious Correlations Experiment
# Replicates Risse et al. (2025) RQ2 on all VulnBench datasets

echo "======================================"
echo "Spurious Correlations Experiment"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ensemble

# Run experiment on all datasets
cd /path/to/VulnBench/scripts  # UPDATE THIS PATH

python spurious_correlations.py \
    --dataset all \
    --project-root .. \
    --tokenizer microsoft/unixcoder-base-nine \
    --seed 42 \
    --output ../results/spurious_correlations_all.json

echo ""
echo "End time: $(date)"
echo "======================================"
