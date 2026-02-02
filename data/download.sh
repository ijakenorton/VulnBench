#!/bin/bash

# Check if hf command exists
if ! command -v hf >/dev/null 2>&1; then
    echo "Hugging Face CLI not found."
    echo "it can be installed now with:"
    echo "    pip install -U \"huggingface_hub[cli]\""
    read -p "Install it now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -U "huggingface_hub[cli]"
    else
        echo "Cannot proceed without HF CLI. Exiting."
        exit 1
    fi
fi

host="ijakenorton"

datasets=(
    "cvefixes"
    "devign"
    "diversevul"
    "draper"
    "icvul"
    "juliet"
    "mvdsc_mixed"
    "reveal"
    "vuldeepecker"
)

for dataset in "${datasets[@]}"; do
    echo "Downloading $dataset..."
    hf download "${host}/${dataset}_for_ml" --repo-type=dataset --force-download --local-dir ./${dataset}/
done
