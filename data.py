import torch
from torch.utils.data import Dataset
import os
import numpy as np

class MNIST(Dataset):

    def __init__(self, root, task):
        if task == 'train':
            self.files = [os.path.join(root, 'train_{}.npz').format(i) for i in range(5)]
        if task == 'test':
            self.files = [os.path.join(root, 'test.npz')]
        self.X, self.Y = self._load_files(self.files)

    def _load_files(self, files):
        X, Y = [], []
        for training_file in files:
            data = np.load(training_file)
            X.append(data['images'])
            Y.append(data['labels'])

        X = np.concatenate(X, axis=0)
        X = torch.tensor(X, dtype=torch.float32)
        X = X.unsqueeze(1)

        Y = np.concatenate(Y, axis=0)
        Y = torch.tensor(Y)

        return X, Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.X.shape[0]
