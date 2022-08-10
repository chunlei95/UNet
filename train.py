import random
from argparse import ArgumentParser
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import transforms
from data import CustomDataset, load_data
from model import UNet

args_parser = ArgumentParser()
args_parser.add_argument('-lr', type=float, default=1e-4)
args_parser.add_argument('-epoch', type=int, default=20)
args_parser.add_argument('-num_classes', type=int, default=2)
args = args_parser.parse_args()


# noinspection PyShadowingNames,SpellCheckingInspection
def train(train_loader, valid_loader, model, criterion, optimizer, total_epoch, current_epoch=0, num_classes=2,
          device='cpu'):
    model.to(device)
    criterion.to(device)
    loss_change_list = []
    valid_loss_change_list = []
    saved_last = {}
    saved_best = {}
    loss_change = {}

    search_best = SearchBest()

    for i in range(current_epoch, total_epoch):
        model.train()
        total_loss = 0.0
        for index, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            y_onehot = convert_to_one_hot(y, num_classes=num_classes)
            predict = model(x)

            loss_value = criterion(predict, y_onehot)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()
            print('Epoch {}: Batch {}/{} loss: {:.4f}'.format(i + 1, index + 1, len(train_loader), loss_value.item()))

        loss_change_list.append(total_loss / len(train_loader))
        valid_avg_loss = valid(model, criterion, valid_loader, num_classes, device)
        valid_loss_change_list.append(valid_avg_loss)
        print('Epoch {} train loss: {:.4f} valid loss: {:.4f}'.format(i + 1, total_loss / len(train_loader),
                                                                      valid_avg_loss))
        search_best(valid_avg_loss)
        if search_best.counter == 0:
            # save the relevant params of the best model state in the current time.
            saved_best['best_model_state_dict'] = model.state_dict()
            saved_best['best_optimizer_state_dict'] = optimizer.state_dict()
            saved_best['epoch'] = i + 1
    loss_change['train_loss_change_history'] = loss_change_list
    loss_change['valid_loss_change_history'] = valid_loss_change_list
    saved_last['last_model_state_dict'] = model.state_dict()
    saved_last['last_optimizer_state_dict'] = optimizer.state_dict()
    saved_last['epoch'] = total_epoch
    torch.save(saved_best, './best_model.pth')
    torch.save(saved_last, './last_model.pth')
    torch.save(loss_change, './loss_change.pth')


def convert_to_one_hot(data, num_classes):
    if type(data) is not torch.Tensor:
        raise RuntimeError('data must be a torch.Tensor')
    if data.dtype is not torch.int64:
        data = data.to(torch.int64)
    data = F.one_hot(data, num_classes=num_classes).permute((0, -1, 1, 2))
    return data.to(torch.float32)


class SearchBest(object):
    def __init__(self, min_delta=0, verbose=True):
        super(SearchBest, self).__init__()
        self.counter = 0
        self.min_delta = min_delta
        self.best_score = None
        self.verbose = verbose

    def __call__(self, valid_loss):
        if self.best_score is None:
            self.best_score = valid_loss
        elif self.best_score - valid_loss >= self.min_delta:
            self.best_score = valid_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print('performance reducing: {}'.format(self.counter))


# noinspection PyShadowingNames
def valid(model, criterion, valid_loader, num_classes, device):
    """
    :return: validate loss
    """
    model.eval()
    valid_total_loss = 0.0
    for index, (x, y) in enumerate(valid_loader):
        x = x.to(device)
        y = y.to(device)
        y_onehot = convert_to_one_hot(y, num_classes)
        with torch.no_grad():
            predict = model(x)
            valid_loss = criterion(predict, y_onehot)
            valid_total_loss += valid_loss.item()
    return valid_total_loss / len(valid_loader)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    root_path = glob('./data/membrane/train/aug/*')
    image_path = [path for path in root_path if path.find('image') != -1]
    target_path = [path for path in root_path if path.find('mask') != -1]
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(),  # 这个貌似有问题
        # transforms.GrayScale(),  # 这个貌似也有问题
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    data_length = len(image_path)
    validation_size = int(data_length * 0.2)
    np.random.seed(42)
    np.random.shuffle(image_path)
    np.random.shuffle(target_path)
    train_image_path = image_path[:data_length - validation_size]
    train_mask_path = target_path[:data_length - validation_size]
    valid_image_path = image_path[data_length - validation_size:]
    valid_mask_path = target_path[data_length - validation_size:]

    train_loader = load_data(train_image_path, train_mask_path, batch_size=2, drop_last=True,
                             transforms=train_transforms)
    valid_loader = load_data(valid_image_path, valid_mask_path, batch_size=2, transforms=valid_transforms)

    train(train_loader, valid_loader, model, criterion, optimizer, args.epoch, device=device)
