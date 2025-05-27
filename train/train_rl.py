import os
import json
import multiprocessing
from pytorch_lightning import Trainer
from scstLightning import SCSTLightningModule
from callbacks.scstloggingcallback import SCSTLoggingCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from mydatasets.mimic_dataset import mimic_Dataset
from torch.utils.data import DataLoader
import argparse

####################################################################
# model
####################################################################
parser = argparse.ArgumentParser(description='Train NLL for RRG.')

# Agrega argumentos posibles
parser.add_argument('--exp_name', type=str, help='Experiment name.')
parser.add_argument('--cluster_id', type=str, required=True, help='Cluster ID.')
parser.add_argument('--process_id', type=str, required=True, help='Process ID.')
parser.add_argument('--model_arch', type=str, help='Architecture to train')
parser.add_argument('--load_weights', type=str, default=None, help='Load weights.')
parser.add_argument('--hnm', type=bool, default=False, help='Use Hard Negative Mining.')

# RL args
parser.add_argument('--scores', type=str, default=None, help='Load weights.')
parser.add_argument('--scores_args', type=str, default=None, help='Load weights.')
parser.add_argument('--scores_weights', type=str, default=None, help='Load weights.')
parser.add_argument('--use_nll', type=bool, default=True, help='Use NLL in SCST.')
parser.add_argument('--top_k', type=int, default=0, help='top_k value.')
parser.add_argument('--resume_ckpt', type=str, default=None, help='Path to checkpoint to resume training from.')

# Parsea los argumentos
args = parser.parse_args()

# Prepare RL args
scores = args.scores.split(":")

scores_weights = args.scores_weights.split(':')
for i in range(len(scores_weights)):
    scores_weights[i] = float(scores_weights[i])

scores_args = args.scores_args.split(":")
for i in range(len(scores_args)):
    scores_args[i] = json.loads(scores_args[i])

experiment_path = os.path.join("../EXPERIMENTS/", args.exp_name)
os.makedirs(experiment_path, exist_ok=True)

####################################################################
# model
####################################################################

load_weights = None if args.resume_ckpt else 'SwinBERT_UnfreezedEncoder/best_meteor_3_model.pt'


model = SCSTLightningModule(
    model_arch='SwinBERTFinetuned',
    load_weights=load_weights,
    scores=scores,
    scores_args=scores_args,
    scores_weights=scores_weights,
    use_nll=True,
    top_k=0,
    lr=5e-5,
    log_dir=experiment_path 
)

####################################################################
# mimic_Dataset
####################################################################


test_dataset = mimic_Dataset(
                transform=model.val_transform, 
                tokenizer=model.tokenizer,
                processor=model.processor,
                partition = "test",
                multi_image=3
                )

train_dataset = mimic_Dataset(
                transform=model.train_transform, 
                tokenizer=model.tokenizer,
                processor=model.processor,
                partition = "train",
                multi_image=3
                )

####################################################################
# Dataloader
####################################################################


batch_size = 1
accumulate_grad_batches = 12
num_workers = int(os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count() - 1))
print("Num workers", num_workers)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    collate_fn=train_dataset.get_collate_fn())

test_dataloader = DataLoader(
    test_dataset, 
    1, 
    shuffle=False, 
    num_workers=num_workers,
    collate_fn=test_dataset.get_collate_fn())


####################################################################
# callbacks
####################################################################


logging_callback = SCSTLoggingCallback(log_dir=experiment_path)

checkpoint_callback = ModelCheckpoint(
    dirpath=experiment_path,
    filename="{epoch:02d}-{step}",
    save_top_k=-1,                            # <- keep all checkpoints (optional, adjust as needed)
    every_n_epochs=1,                         # <- save once per epoch
    save_on_train_epoch_end=True,
)

####################################################################
# train
####################################################################
if args.resume_ckpt:
    resume_checkpoint_path = os.path.join("../EXPERIMENTS/", args.resume_ckpt)
else:
    resume_checkpoint_path = None


trainer = Trainer(
    max_epochs=50,
    default_root_dir=experiment_path,
    callbacks=[logging_callback, checkpoint_callback],
)
trainer.fit(model, train_dataloader, test_dataloader, ckpt_path=resume_checkpoint_path)
