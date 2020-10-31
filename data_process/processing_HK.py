import h5py
from PIL import Image
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:

        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):

        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        return img


class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return img


class RandomGaussianBlur(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, img):

        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return img


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img


def transform_tr(sample):
    composed_transforms = transforms.Compose([
        # RandomHorizontalFlip(),
        # RandomScaleCrop(base_size=224, crop_size=224),
        # RandomGaussianBlur(),
        FixScaleCrop(crop_size=224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])

    return composed_transforms(sample)


def transform_val(sample):
    composed_transforms = transforms.Compose([
        FixScaleCrop(crop_size=224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])
    return composed_transforms(sample)


def read_data():
    c1 = np.load('/home1/herundong1/Downloads/liuxiangfei/Aes_Seg/data_process/list_index_HK.npy')
    c2 = np.load('/home1/herundong1/Downloads/liuxiangfei/Aes_Seg/data_process/list_aes_HK.npy')
    c3 = np.load('/home1/herundong1/Downloads/liuxiangfei/Aes_Seg/data_process/list_seg_HK.npy')

    train_path_decode = list()
    train_label_aes = list()
    train_label_seg = list()
    val_path_decode = list()
    val_label_aes = list()
    val_label_seg = list()

    for i in range(7000):
        train_path_decode.append(c1[i])
        train_label_aes.append(c2[i])
        train_label_seg.append(c3[i])
    for i in range(7000, 10000):
        val_path_decode.append(c1[i])
        val_label_aes.append(c2[i])
        val_label_seg.append(c3[i])

    return train_path_decode, train_label_aes, train_label_seg, val_path_decode, val_label_aes, val_label_seg


class MyDataset(data.Dataset):

    def __init__(self, train=True):
        if train:
            self.path, self.label_gender, self.label_age, _, _, _ = read_data()
        else:
            _, _, _, self.path, self.label_gender, self.label_age = read_data()
        self.train = train

    def __getitem__(self, index):
      #  print(self.path[index])
        image = Image.open(self.path[index]).convert('RGB')
#        image = Image.open('/home/liuxiangfei/PhotoQualityDataset/' + self.path[index] + '.jpg').convert('RGB')

        if self.train:
            image = transform_tr(image)
            return image, int(self.label_gender[index]), self.label_age[index]
        else:
            image = transform_val(image)
            return image, int(self.label_gender[index]), self.label_age[index]

    def __len__(self):
        return len(self.path)


def make_loader():
    train_data = MyDataset(train=True)
    val_data = MyDataset(train=False)
    trainloader = data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    valloader = data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    return train_data, val_data, trainloader, valloader


# c = np.load('hotdata.npy')
# print(c[:9893].shape[0], c[:9893].shape[1])
# path = '/home1/shenzhen/faces/'
# print(c.shape)
# for i in range(10):
#     print(path + c[i][0], c[i][-2], c[i][-1])
#     img = Image.open(path + c[i][0])
#     plt.imshow(img)
#     plt.show()

# train_data, val_data, trainloader, valloader = make_loader()
# #
# print(len(train_data))   # 4000
# print(len(val_data))     # 3000
# print(len(trainloader))  # 125
# print(len(valloader))    # 94

# train_data = MyDataset(train=False)
# # # 953980 440774 954113 953958 953619 953349 954175 953897 310261 953841 179118 371434 848725 delta=0
# # # 567829 277832
# #
# for i in range(7000):
#     print(i)
#     # if i in l:
#     #     continue
#     a,b,c = train_data.__getitem__(i)