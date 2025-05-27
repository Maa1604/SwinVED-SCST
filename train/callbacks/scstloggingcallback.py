import os
import csv
from pytorch_lightning.callbacks import Callback
from pycocoevalcap.metrics import Evaluator
import torch


class SCSTLoggingCallback(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path_txt = os.path.join(self.log_dir, "log.txt")
        self.log_path_csv = os.path.join(self.log_dir, "captions.csv")

    def on_validation_epoch_end(self, trainer, pl_module):
        gts = {k: [{'caption': v}] for k, v in enumerate(pl_module.l_refs)}
        res = {k: [{'caption': v}] for k, v in enumerate(pl_module.l_hyps)}

        evaluator = Evaluator()
        evaluator.do_the_thing(gts, res)

        train_loss = trainer.logged_metrics.get('train_loss_epoch', 0.0)
        val_loss = trainer.logged_metrics.get('val_loss', 0.0)

        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.cpu().item()
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.cpu().item()

        metrics_table = Evaluator.metrics_to_log(
            evaluator.evaluation_report,
            train_loss=train_loss,
            test_loss=val_loss,
            lr=pl_module.lr
        )

        # Write metrics to log.txt
        with open(self.log_path_txt, "a") as f:
            f.write(f"\nepoch {trainer.current_epoch}:\n")
            f.write(metrics_table + "\n")

        # Save captions to uniquely named CSV
        csv_name = f"epoch{trainer.current_epoch:02d}-step{trainer.global_step}.csv"
        csv_path = os.path.join(self.log_dir, csv_name)

        with open(csv_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["reference", "hypothesis"])
            for ref, hyp in zip(pl_module.l_refs, pl_module.l_hyps):
                writer.writerow([ref, hyp])

        # Clear lists
        pl_module.l_refs.clear()
        pl_module.l_hyps.clear()
