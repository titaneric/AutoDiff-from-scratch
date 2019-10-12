import random

class Dataset:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class DataLoader:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def next_batch(self, num):
        candidate = list(range(len(self.dataset)))
        sample_index = random.sample(candidate, num)
        return self.dataset[sample_index]