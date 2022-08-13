from glob import glob

import torch
from ignite.metrics import Precision, Recall, DiceCoefficient, ConfusionMatrix, mIoU

import evaluate
import transforms
from data import load_data, CustomDataset
from model import UNet
import matplotlib.pyplot as plt


# noinspection PyShadowingNames
def predict(model, test_loader, device):
    precision_ignite = Precision(average=True)
    recall_ignite = Recall(average=True)
    confusion_matrix = ConfusionMatrix(num_classes=2)
    dice_coef_ignite = DiceCoefficient(confusion_matrix)
    miou = mIoU(confusion_matrix)

    if model.training:
        model.eval()

    predict_y, real_y = torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
    with torch.no_grad():
        for index, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y_predict = model(x)

            predict_class = torch.max(y_predict, dim=1)[1]
            predict_y = torch.cat([predict_y, predict_class])
            real_y = torch.cat([real_y, y])

            confusion_matrix.update((y_predict, y))
            precision_ignite.update((predict_class, y))
            recall_ignite.update((y_predict, y))
            dice_coef_ignite.update((y_predict, y))
            miou.update((y_predict, y))

    dice = evaluate.dice_coef(predict_y, real_y)
    precision = evaluate.precision(predict_y, real_y)
    recall = evaluate.recall(predict_y, real_y)

    print(dice)
    print(precision)
    print(recall)

    print('confusion matrix: {}'.format(confusion_matrix.compute()))
    print('precision: {}'.format(precision_ignite.compute()))
    print('recall: {}'.format(recall_ignite.compute()))
    print('dice coef: {}'.format(dice_coef_ignite.compute()))
    print('miou: {}'.format(miou.compute()))


def plot_image(image):
    pass


def plot_loss_change(train_losses, val_losses=None):
    figure, axes = plt.subplots()
    plt.plot(range(len(train_losses)), train_losses)
    if val_losses is not None:
        plt.plot(range(len(val_losses)), val_losses)
        plt.legend(['train loss', 'val loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.show()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    dataset = CustomDataset
    test_image_path = glob('./dataset/test/images/*')
    test_mask_path = glob('./dataset/test/masks/*')

    input_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_loader = load_data(data_path=test_image_path, target_path=test_mask_path,
                            batch_size=32, drop_last=False,
                            transforms=input_transforms)
    model = UNet()
    trained_params = torch.load('./pretrained/best_model.pth', map_location=device)
    model.load_state_dict(trained_params['best_model_state_dict'])
    model.eval()

    predict(model, test_loader, device)
