# conda activate hroc

python mytrain_rl_steps.py \
    --exp_name SwinBERTFinetuned_RLpruebaConBertscore4singularityyyy \
    --model_arch SwinBERTFinetuned \
    --load_weights SwinBERT_UnfreezedEncoder/best_meteor_3_model.pt \
    --scores_weights 0.6,0.4 \
    --scores BertScorer \
    --scores_args {} \
    --use_nll True \
    --top_k 0

