import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data_path, target_path, transforms=None):
        super(CustomDataset, self).__init__()
        self.data_paths = data_path
        self.target_paths = target_path
        self.transforms = transforms

    def __getitem__(self, item):
        image = plt.imread(self.data_paths[item])
        target = plt.imread(self.target_paths[item])

        image = np.expand_dims(image, axis=-1)

        # image = torch.from_numpy(image)
        # target = torch.from_numpy(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.data_paths)


def load_data(data_path, target_path, batch_size, drop_last=False, transforms=None):
    datas = CustomDataset(data_path=data_path, target_path=target_path, transforms=transforms)
    data_loader = DataLoader(datas, shuffle=True, batch_size=batch_size, drop_last=drop_last)
    return data_loader
