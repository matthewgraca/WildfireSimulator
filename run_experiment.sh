#!/bin/bash
#experiment_dir="/mnt/wildfire/surrogate-model/2026-08-21-fat-quantized"
#samples=100

while getopts "d:s:" opt; do
    case $opt in
        d) EXPERIMENT_DIR="$OPTARG" ;;
        s) SAMPLES="$OPTARG" ;;
        *) echo "Usage: $0 [-d experiment directory] [-s samples]"; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT_DIR" || -z "$SAMPLES" ]]; then
    echo "Usage: $0 [-d experiment directory] [-s samples]"
    exit 1
fi

echo "Experiment directory set to: $EXPERIMENT_DIR"
echo "Number of samples set to: $SAMPLES"

python scripts/run.py --output-dir $experiment_dir && \
python scripts/evaluate.py $experiment_dir --num_samples $samples && \
python scripts/ablation.py $experiment_dir --num_samples $samples 
