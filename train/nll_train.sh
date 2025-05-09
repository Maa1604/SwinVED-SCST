# conda activate hroc

python mytrain_nll.py \
    --exp_name SwinBERT_UnfreezedEncoder \
    --model_arch SwinBERTFinetuned \
    --hnm True \
    --load_weights "SwinBERT_FreezedEncoder/best_bleu1_6_model.pt"
    # --metrics_on_train