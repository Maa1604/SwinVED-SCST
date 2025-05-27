#!/bin/bash
cd /home/maasala/llama-vqa/train 

nvidia-smi --query-gpu=name --format=csv,noheader

echo "RUNNING: $0 $@"

CLUSTER_ID=$1
PROCESS_ID=$2
SCORES_WEIGHTS=$3
SCORES=$4
SCORES_ARGS=$5
CHKP=$6

# Sanitize variables for use in tag name
TAG_NAME="run_${CLUSTER_ID}_${PROCESS_ID}"

# Create Git tag with metadata in the message
git tag -a $TAG_NAME -m "Run with:
- Cluster ID: $CLUSTER_ID
- Process ID: $PROCESS_ID
- Scores: NLL,$SCORES
- Scores Weights: $SCORES_WEIGHTS
- Scores Args: $SCORES_ARGS"

git push origin $TAG_NAME


# Build pretty score string: nll weight + each scorer and weight
IFS=':' read -r -a WEIGHTS_ARRAY <<< "$SCORES_WEIGHTS"
IFS=':' read -r -a SCORES_ARRAY <<< "$SCORES"

SCORE_STRING="nll:${WEIGHTS_ARRAY[0]}"
for i in "${!SCORES_ARRAY[@]}"; do
    SCORE_STRING+=",${SCORES_ARRAY[$i]}:${WEIGHTS_ARRAY[$((i + 1))]}"
done

# Store the full experiment name in a variable and wrap it in quotes
EXP_NAME="SwinBERTFinetuned_RL_${CLUSTER_ID}.${PROCESS_ID}__Scores__${SCORE_STRING}"

# (Optional) Use them in your script
echo "Running Cluster ID: $CLUSTER_ID, Process ID: $PROCESS_ID"

# Optional argument for checkpoint
RESUME_ARG=""
if [ -n "$CHKP" ]; then
    RESUME_ARG="--resume_ckpt $CHKP"
fi


python train_rl.py \
    --exp_name "$EXP_NAME" \
    --cluster_id ${CLUSTER_ID} \
    --process_id ${PROCESS_ID} \
    --model_arch SwinBERTFinetuned \
    --load_weights SwinBERT_UnfreezedEncoder/best_meteor_3_model.pt \
    --scores_weights ${SCORES_WEIGHTS} \
    --scores ${SCORES} \
    --scores_args ${SCORES_ARGS} \
    --use_nll True \
    --top_k 0 \
    $RESUME_ARG

