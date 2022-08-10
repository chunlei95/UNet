import random

import transforms
from PIL import Image
from glob import glob


class Generator(object):

    def __init__(self, counts, sources_target_path, sources_mask_path, transforms):
        super(Generator, self).__init__()
        self.counts = counts
        self.sources_path = sources_target_path
        self.sources_mask = sources_mask_path
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        for index, (image, target) in enumerate(zip(self.sources_path, self.sources_mask)):
            for i in range(self.counts):
                source_image = Image.open(image)
                source_target = Image.open(target)
                if self.transforms is not None:
                    source_image, source_target = self.transforms(source_image, source_target)
                source_image.save(
                    './data/membrane/train/aug/' + 'image_new' + str(index) + '_' + str(
                        random.randint(1000000, 9999999)) + '.png')
                source_target.save(
                    './data/membrane/train/aug/' + 'mask_new' + str(index) + '_' + str(
                        random.randint(1000000, 9999999)) + '.png')


if __name__ == '__main__':
    root_path = glob('./data/membrane/train/aug/*')
    source_image_path = [path for path in root_path if path.find('image') != -1]
    source_target_path = [path for path in root_path if path.find('mask') != -1]
    count = 20
    generator_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((0, 180)),
        transforms.ColorJitter(),
        transforms.GrayScale(),
        transforms.RandomCrop(256),
    ])
    generator = Generator(count, source_image_path, source_target_path, generator_transforms)
    generator()
