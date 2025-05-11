#!/bin/bash

echo "RUNNING: $0 $@"

CLUSTER_ID=$1
PROCESS_ID=$2
SCORES_WEIGHTS=$3
SCORES=$4
SCORES_ARGS=$5

# (Optional) Use them in your script
echo "Running Cluster ID: $CLUSTER_ID, Process ID: $PROCESS_ID"

python mytrain_rl_steps.py \
    --exp_name SwinBERTFinetuned_RL_${CLUSTER_ID}_${PROCESS_ID}__Scores__NLL,${SCORES}__Scores_weights__${SCORES_WEIGHTS} \
    --cluster_id ${CLUSTER_ID} \
    --process_id ${PROCESS_ID} \
    --model_arch SwinBERTFinetuned \
    --load_weights SwinBERT_UnfreezedEncoder/best_meteor_3_model.pt \
    --scores_weights ${SCORES_WEIGHTS} \
    --scores ${SCORES} \
    --scores_args ${SCORES_ARGS} \
    --use_nll True \
    --top_k 0

