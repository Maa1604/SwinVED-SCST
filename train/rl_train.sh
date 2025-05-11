#!/bin/bash

CLUSTER_ID=$1
PROCESS_ID=$2

# (Optional) Use them in your script
echo "Running Cluster ID: $CLUSTER_ID, Process ID: $PROCESS_ID"

python mytrain_rl_steps.py \
    --exp_name SwinBERTFinetuned_RL_${CLUSTER_ID}_${PROCESS_ID}_scores_scores_weights \
    --cluster_id ${CLUSTER_ID} \
    --process_id ${PROCESS_ID} \
    --model_arch SwinBERTFinetuned \
    --load_weights SwinBERT_UnfreezedEncoder/best_meteor_3_model.pt \
    --scores_weights 0.6,0.4 \
    --scores BertScorer \
    --scores_args {} \
    --use_nll True \
    --top_k 0

