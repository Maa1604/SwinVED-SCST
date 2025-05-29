import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))) # "/home/user/RRG/rrg"


from mymodels.swinbertcrossLORA import SwinBERTFinetuned
from myrl.scst import SCST

class SCSTLightningModule(pl.LightningModule):
    def __init__(self, model_arch, load_weights=None, scores=None, scores_args=None, scores_weights=None,
                 use_nll=True, top_k=0, lr=5e-5, log_dir=None):
        super().__init__()

        self.save_hyperparameters()

        self.model = {
            "SwinBERTFinetuned": SwinBERTFinetuned(),
        }[model_arch]

        if load_weights:
            actual_weights_path = os.path.join("../EXPERIMENTS/", load_weights)
            state_dict = torch.load(actual_weights_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {actual_weights_path}")
        else:
            print("No manual weight loading (possibly resuming from checkpoint)")

        self.scst = SCST(
            bos_token_id=self.model.bos_token_id,
            eos_token_id=self.model.eos_token_id,
            pad_token_id=self.model.pad_token_id,
            scores=scores,
            scores_args=scores_args,
            scores_weights=scores_weights,
            use_nll=use_nll,
            top_k=top_k
        )

        self.log_dir = log_dir
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = os.path.join(self.log_dir, "log.txt")
        else:
            self.log_path = None



        self.val_transform = self.model.val_transform
        self.train_transform = self.model.train_transform
        self.tokenizer = self.model.tokenizer
        self.processor = self.model.processor
        self.criterion = nn.NLLLoss()
        self.lr = lr
        
        self.l_refs = []
        self.l_hyps = []

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        pixel_values = batch['images']
        questions_ids = batch['questions_ids']
        questions_mask = batch['questions_mask']
        answers_ids = batch['answers_ids']
        answers_mask = batch['answers_mask']
        images_mask = batch['images_mask']
        labels = batch['labels']

        with torch.no_grad():
            self.model.eval()
            out_greedy = self.scst.forward_greedy(
                questions_ids=questions_ids,
                questions_mask=questions_mask,
                answers_ids=answers_ids,
                answers_mask=answers_mask,
                images=pixel_values,
                model=self.model,
                images_mask=images_mask
            )

        self.model.train()
        out = self.scst.forward_sampling(
            questions_ids=questions_ids,
            questions_mask=questions_mask,
            answers_ids=answers_ids,
            answers_mask=answers_mask,
            reward_greedy=out_greedy["reward_greedy"],
            images=pixel_values,
            model=self.model,
            images_mask=images_mask,
            labels=labels
        )

        self.log('train_loss', out["nll_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        pixel_values = batch['images']
        questions_ids = batch['questions_ids']
        questions_mask = batch['questions_mask']
        answers_ids = batch['answers_ids']
        answers_mask = batch['answers_mask']
        images_mask = batch['images_mask']
        labels = batch['labels']

        output = self.model(
            questions_ids=questions_ids,
            questions_mask=questions_mask,
            answers_ids=answers_ids,
            answers_mask=answers_mask,
            images=pixel_values,
            images_mask=images_mask,
            labels=labels
        )

        generated_answers, _ = self.model.generate(
            pixel_values,
            images_mask=images_mask,
            questions_ids=questions_ids,
            questions_mask=questions_mask,
            tokenizer=self.model.tokenizer,
            num_beams=2,
            max_len=128,
            return_dict_in_generate=True,
            output_scores=True
        )

        references = batch['answers']

        self.l_refs.extend(references)
        self.l_hyps.extend(generated_answers)

        self.log('val_loss', output['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #return output['loss']
    def on_validation_epoch_end(self):
        
        self.l_refs.clear()
        self.l_hyps.clear()