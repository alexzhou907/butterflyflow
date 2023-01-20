import os
import scipy.io
import numpy as np

import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

import math

from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from .patient import MIMIC


def load_cifar10(data_path):
    imageSize = 32
    train_data = datasets.CIFAR10(data_path, train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.Pad(4, padding_mode='reflect'),
                                      transforms.RandomCrop(imageSize),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.ToTensor()
                                  ]))
    test_data = datasets.CIFAR10(data_path, train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor()
                                 ]))
    return train_data, test_data


def load_mimic(data_path, patient):
    imageSize = 32
    train_data = MIMIC(os.path.join(data_path, patient), time_len=1024, train=True)
    test_data = MIMIC(os.path.join(data_path, patient), time_len=1024, train=False)
    return train_data, test_data


def load_galaxy(data_path, imageSize):
    """
    assumes that data has been preprocessed into .pkl files using prepare_galaxy_dataset.py in caterpillar_flow/ directory.
    images are 64 x 64, downsampling currently not supported
    """
    # uses data augmentation for training
    # https://github.com/ehoogeboom/emerging/blob/master/data_loaders/get_mnist_cifar.py

    # ToPILImage transformation is really annoying, but we already have preprocessed pytorch tensors 
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
        random_transpose
    ])

    train_data = GalaxyDataset(root_dir=data_path, split='train', resolution=imageSize, transform=train_transform)
    val_data = GalaxyDataset(root_dir=data_path, split='valid', resolution=imageSize, transform=None)

    return train_data, val_data




def get_batch(data, indices):
    imgs = []
    labels = []
    for index in indices:
        img, label = data[index]
        imgs.append(img)
        labels.append(label)
    return torch.stack(imgs, dim=0), torch.LongTensor(labels)


def iterate_minibatches(data, indices, batch_size, shuffle):
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(indices), batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield get_batch(data, excerpt)


def binarize_image(img):
    return torch.rand(img.size()).type_as(img).le(img).float()


def binarize_data(data):
    return [(binarize_image(img), label) for img, label in data]


def preprocess(img, n_bits, noise=None):
    n_bins = 2. ** n_bits
    # rescale to 255
    img = img.mul(255)
    if n_bits < 8:
        img = torch.floor(img.div(256. / n_bins))

    if noise is not None:
        # [batch, nsamples, channels, H, W]
        img = img.unsqueeze(1) + noise
    # normalize
    img = img.div(n_bins)
    img = (img - 0.5).div(0.5)
    return img

def permute_block(img, perm):

    b,c,h,w = img.shape
    img = img.reshape(b,-1)
    return img[:,perm].reshape(b,c,h,w)

def get_permute_matrix(img, level, tp=0):
    b,c,h,w = img.shape
    data_dim = c*h*w
    assert data_dim % (2 ** (level + 1)) == 0
    # img = img.reshape(b,-1)

    l = data_dim // (2 ** (level))

    if tp == 0:
        perm = torch.cat([torch.cat([torch.arange(l//2, l) + i*l, torch.arange(0, l//2)+ i*l]) for i in range(2**level)])
        #[[A,B],[C,D]] where A,D are 0, B,C are 1 diagonal
    elif tp == 1:

        perm = []
        for i in range(2**level):
            # ind_00 = torch.arange(0, l//2, 2) + i*l
            # ind_01 =  torch.arange(1, l//2, 2) + i*l + l//2
            # ind_10 = torch.arange(1, l // 2, 2)+ i*l
            # ind_11 =  torch.arange(0, l // 2, 2)+ i*l + l//2

            ind_0 = []
            ind_1 = []

            for j in range(l//2):
                if j % 2 == 0:
                    ind_0.append(j+ i*l)
                else:
                    ind_0.append(j + i*l  + l//2)
            for j in range(l//2):
                if j % 2 == 1:
                    ind_1.append(j+ i*l )
                else:
                    ind_1.append(j + i*l + l//2)

            perm.append(torch.cat([torch.tensor(ind_0), torch.tensor(ind_1)]))

        perm = torch.cat(perm)
    elif tp == 2:
        perm = []
        for i in range(2 ** level):
            # ind_00 = torch.arange(0, l//2, 2) + i*l
            # ind_01 =  torch.arange(1, l//2, 2) + i*l + l//2
            # ind_10 = torch.arange(1, l // 2, 2)+ i*l
            # ind_11 =  torch.arange(0, l // 2, 2)+ i*l + l//2

            ind_0 = []
            ind_1 = []

            for j in range(l // 2):
                a = np.random.choice(2)
                if a == 0:
                    ind_0.append(j+ i * l)
                else:
                    ind_0.append(j+ i * l+ l // 2)


            for j in range(l // 2):
                if not (j+ i * l in ind_0) and not (j+ i * l + l//2 in ind_0):


                    a = np.random.choice(2)
                    if a == 0:
                        ind_1.append(j+ i * l)
                    else:
                        ind_1.append(j + i * l + l // 2)
                elif not (j+ i * l in ind_0):
                    ind_1.append(j + i * l)
                elif not (j+ i * l + l//2 in ind_0):
                    ind_1.append(j + i * l + l // 2)


            perm.append(torch.tensor(ind_0+ind_1))

        perm = torch.cat(perm)
    else:
        raise NotImplementedError

    return (perm, l)

def get_permute_all(img, layer_num=8, tp=0):
    pp = []
    ls = []
    for i in range(layer_num):
        perm, l = get_permute_matrix(img, i, tp)
        pp.append(perm)
        ls.append(l)
    return pp, ls

def permute(img, perm_all):
    for perm in perm_all:
        img = permute_block(img, perm)
    return img

def postprocess(img, n_bits):
    n_bins = 2. ** n_bits
    # re-normalize
    img = img.mul(0.5) + 0.5
    img = img.mul(n_bins)
    # scale
    img = torch.floor(img) * (256. / n_bins)
    img = img.clamp(0, 255).div(255)
    return img


def logit_transform( image, rescale=True, lambd=0.000001):
    if rescale:
        image = (image*0.5)+0.5
    image = lambd + (1 - 2 * lambd) * image
    return torch.log(image) - torch.log1p(-image)

def sigmoid_transform(samples, rescale=True, lambd=0.000001):
    samples = torch.sigmoid(samples)
    samples = (samples - lambd) / (1 - 2 * lambd)

    if rescale:
        samples = (samples-0.5)/0.5
    return samples
