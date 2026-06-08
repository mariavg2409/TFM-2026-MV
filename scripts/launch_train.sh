#!/bin/bash

# a fixed seed ensures KFold produces identical splits across all runs
# SEED=42
KFOLDS=10

declare -A RUN_NAMES=(
    ["data/emb_diseases.npy"]="ours_m1"
    ["data/emb_diseases_m2.npy"]="ours_m2"
)

for emb_path in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$emb_path]}"

    python train.py config/train_delphi_demo.py \
        --device=cuda \
        --wandb_log=True \
        --wandb_run_name="$run_name" \
        --out_dir="out/$run_name" \
        --k_folds=$KFOLDS \
        --embeddings_path="$emb_path" \
        --adapter_n_layers=1 \
        --adapter_n_hidden=128
done


python train.py config/train_delphi_demo.py \
    --device=cuda \
    --wandb_log=True \
    --wandb_run_name=delphi \
    --out_dir="out/delphi" \
    --k_folds=$KFOLDS

