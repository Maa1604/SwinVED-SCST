import random
from torch.utils.data import Sampler

class StatefulSampler(Sampler):
    def __init__(self, data_source, shuffle=True, seed=42):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self.indices = list(range(len(data_source)))
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(self.indices)

    def __iter__(self):
        start = self.position
        self.position = 0
        for i in range(start, len(self.indices)):
            yield self.indices[i]
        self.epoch += 1
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'position': self.position,
            'indices': self.indices
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.position = state_dict['position']
        self.indices = state_dict['indices']
        print(f"[StatefulSampler] Resumed at epoch {self.epoch}, position {self.position}")

