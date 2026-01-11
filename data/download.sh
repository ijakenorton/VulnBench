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
    "${host}/cvefixes_for_ml"
    "${host}/devign_for_ml"
    "${host}/diversevul_for_ml"
    "${host}/draper_for_ml"
    "${host}/icvul_for_ml"
    "${host}/juliet_for_ml"
    "${host}/mvdsc_mixed_for_ml"
    "${host}/reveal_for_ml"
    "${host}/vuldeepecker_for_ml"
)

for dataset in "${datasets[@]}"; do
    echo "Downloading $dataset..."
    hf download "$dataset" --repo-type=dataset --force-download --local-dir .
done
