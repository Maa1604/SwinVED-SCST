from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from mydatasets.mimic_dataset import mimic_Dataset
from stateful_sampler import StatefulSampler
import os
import multiprocessing

SEED = 42

class MimicDataModule(LightningDataModule):
    def __init__(self, model, batch_size=4):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_workers = int(os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count() - 1))

    def setup(self, stage=None):
        self.train_dataset = mimic_Dataset(
            transform=self.model.train_transform,
            tokenizer=self.model.tokenizer,
            processor=self.model.processor,
            partition="train",
            multi_image=3
        )

        self.val_dataset = mimic_Dataset(
            transform=self.model.val_transform,
            tokenizer=self.model.tokenizer,
            processor=self.model.processor,
            partition="test",
            multi_image=3
        )

    def train_dataloader(self):
        sampler = StatefulSampler(self.train_dataset, shuffle=True, seed=SEED)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.get_collate_fn()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.get_collate_fn()
        )
