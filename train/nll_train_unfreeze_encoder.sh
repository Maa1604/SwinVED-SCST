python mytrain_nll.py \
    --exp_name SwinBERT_FrozenEncoder \
    --model_arch SwinVEDFinetuned \
    --hnm True \
    --freeze_encoder \
    --load_weights SwinBERT_UnfreezedEncoder/best_bertscore_0_model.pt