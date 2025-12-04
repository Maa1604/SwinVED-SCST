python mytrain_nll.py \
    --exp_name SwinVED_UnfreezedEncoder \
    --model_arch SwinVEDFinetuned \
    --hnm True \
    --load_weights SwinVED_FrozenEncoder/best_bertscore_0_model.pt