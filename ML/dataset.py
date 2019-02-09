'''
    Extend the torch.utils.data.Dataset class to build a GestureDataset class.
'''
import torch.utils.data as data

class DataServe(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        pass
        return self.x[index], self.y[index]