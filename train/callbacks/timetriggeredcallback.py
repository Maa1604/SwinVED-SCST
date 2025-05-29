import time
import csv
import os
import torch
import wandb
from dotenv import load_dotenv
from pytorch_lightning.callbacks import Callback
from pycocoevalcap.bertscore.bertscore import BertScorer

load_dotenv("../wandb.env")

if not os.getenv("WANDB_API_KEY"):
    raise RuntimeError("WANDB_API_KEY not set. Please use a secure env file.")

class TimeTriggeredCheckpointAndEvalCallback(Callback):
    def __init__(self, log_dir, max_duration_hours=36, args=None):
        super().__init__()
        self.start_time = time.time()
        self.max_duration = max_duration_hours * 3600  # en segundos
        self.log_dir = log_dir
        self.bert_scorer = BertScorer()
        self.triggered = False  # Para evitar múltiples activaciones
        self.l_refs = []
        self.l_hyps = []

        # Parse weights and scores
        scores = args.scores.split(",") if isinstance(args.scores, str) else args.scores
        weights = [float(w) for w in args.scores_weights.split(":")] if isinstance(args.scores_weights, str) else args.scores_weights

        wandb.init(
            project="VQA-RRG-training",
            name=f"{args.cluster_id}.{args.process_id} ; Scores: " +
                 f"nll {weights[0]}" +
                 (", " + ", ".join([f"{s} {w}" for s, w in zip(scores, weights[1:])]) if scores else "")
        )

    def on_train_start(self, trainer, pl_module):
        self.l_refs.clear()
        self.l_hyps.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.triggered:
            return

        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.max_duration:
            self.triggered = True

            # Guarda el checkpoint
            checkpoint_path = f"{self.log_dir}/time_triggered_checkpoint.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            print(f"[Callback] Checkpoint guardado en {checkpoint_path} tras {elapsed_time/3600:.2f} horas.")

            # Evalúa y calcula BERTScore
            pl_module.model.eval()
            self.l_refs.clear()
            self.l_hyps.clear()

            val_dataloader = trainer.val_dataloaders
            device = pl_module.device
            with torch.no_grad():
                for batch in val_dataloader:
                    pixel_values = batch['images'].to(device)
                    questions_ids = batch['questions_ids'].to(device)
                    questions_mask = batch['questions_mask'].to(device)
                    answers_ids = batch['answers_ids'].to(device)
                    answers_mask = batch['answers_mask'].to(device)
                    images_mask = batch['images_mask'].to(device)
                    labels = batch['labels'].to(device)

                    _ = pl_module.model(
                        questions_ids=questions_ids,
                        questions_mask=questions_mask,
                        answers_ids=answers_ids,
                        answers_mask=answers_mask,
                        images=pixel_values,
                        images_mask=images_mask,
                        labels=labels
                    )

                    generated_answers, _ = pl_module.model.generate(
                        pixel_values,
                        images_mask=images_mask,
                        questions_ids=questions_ids,
                        questions_mask=questions_mask,
                        tokenizer=pl_module.model.tokenizer,
                        num_beams=2,
                        max_len=128,
                        return_dict_in_generate=True,
                        output_scores=True
                    )

                    references = batch['answers']
                    self.l_refs.extend(references)
                    self.l_hyps.extend(generated_answers)

                calculated_bertscore = self.bert_scorer(self.l_hyps, self.l_refs)
                print(f"[Callback] BERTScore: {calculated_bertscore}")

                wandb.log({
                        "bertscore": calculated_bertscore
                })

                # Save CSV
                csv_path = checkpoint_path.replace(".ckpt", ".csv")
                with open(csv_path, "w", newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["reference", "hypothesis"])
                    for ref, hyp in zip(self.l_refs, self.l_hyps):
                        writer.writerow([ref, hyp])
                print(f"[Callback] Referencias e hipótesis guardadas en {csv_path}")

                # Clear lists after use
                self.l_refs.clear()
                self.l_hyps.clear()
