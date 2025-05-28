# dataloading.py

import os
import multiprocessing
from torch.utils.data import DataLoader
from mydatasets.mimic_dataset import mimic_Dataset

def get_dataloaders(model, partition="train,test", batch_size=1):
    num_workers = int(os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count() - 1))
    print("Num workers", num_workers)

    train_dataset = mimic_Dataset(
        transform=model.train_transform, 
        tokenizer=model.tokenizer,
        processor=model.processor,
        partition="train",
        multi_image=3
    )

    test_dataset = mimic_Dataset(
        transform=model.val_transform, 
        tokenizer=model.tokenizer,
        processor=model.processor,
        partition="test",
        multi_image=3
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=train_dataset.get_collate_fn()
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=test_dataset.get_collate_fn()
    )

    return train_dataloader, test_dataloader
