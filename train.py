from argparse import ArgumentParser
from glob import glob

import torch
import torch.nn.functional as F
import torch.optim as optim

import transforms
from data import load_data
from loss import DiceLoss
from model import UNet
import torch.nn as nn

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

            # predict_mask = torch.max(predict, dim=1)[1].to(torch.float32)
            # y = y.to(torch.float32)

            loss_value = criterion(predict, y)
            loss_value.requires_grad_(True)
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
        # y_onehot = convert_to_one_hot(y, num_classes)
        with torch.no_grad():
            predict = model(x)
            valid_loss = criterion(predict, y)
            valid_total_loss += valid_loss.item()
    return valid_total_loss / len(valid_loader)


if __name__ == '__main__':
    lr = 3e-4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    # 后面修改为Dice损失函数看看效果
    criterion = nn.CrossEntropyLoss()

    train_image_path = glob('./dataset/train/images/*')
    train_mask_path = glob('./dataset/train/masks/*')

    val_image_path = glob('./dataset/val/images/*')
    val_mask_path = glob('./dataset/val/masks/*')

    train_transforms = transforms.Compose([
        # todo 先Resize试一下
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0, 180)),
        #         ColorJitter(), # 这个有问题，但是是什么问题？
        #         GrayScale(), # 这个貌似也有问题，问题更大
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    valid_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = load_data(train_image_path, train_mask_path, batch_size=16, drop_last=True,
                             transforms=train_transforms)
    valid_loader = load_data(val_image_path, val_mask_path, batch_size=16, transforms=valid_transforms)

    continue_train = False

    if not continue_train:
        epoch = 180
        train(train_loader, valid_loader, model, criterion, optimizer, epoch, device=device)
    else:
        total_epoch = 200
        pretrain_params = torch.load('../input/covid-xray-unet/last_model.pth')
        model.load_state_dict(pretrain_params['last_model_state_dict'])
        optimizer.load_state_dict(pretrain_params['last_optimizer_state_dict'])
        current_epoch = pretrain_params['epoch']
        model.train()
        train(train_loader, valid_loader, model, criterion, optimizer, total_epoch, current_epoch, device=device)
