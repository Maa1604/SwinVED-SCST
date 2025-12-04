python mytrain_rl.py \
    --exp_name exp1_stage2 \
    --model_arch SwinVEDFinetuned \
    --load_weights SwinBERT_UnfreezedEncoder/best_bertscore_0_model.pt\
    --scores BertScorer \
    --scores_args {} \
    --scores_weights 0.4,0.6 \
    --use_nll True \
    --top_k 0 \
