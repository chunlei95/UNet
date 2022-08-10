import torch.nn as nn

from evaluate import dice_coef, IoU


class DiceLoss(nn.Module):
    """Dice相似系数损失函数，训练过程可能出现不稳定的情况

    """

    def __init__(self, ep=1e-8):
        super(DiceLoss, self).__init__()
        self.epsilon = ep

    def forward(self, predict, target):
        score = dice_coef(predict, target, self.epsilon)
        return 1 - score


class IoULoss(nn.Module):
    """IoU损失函数

    """
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, predict, target):
        return IoU(predict, target)
