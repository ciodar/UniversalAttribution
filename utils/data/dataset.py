import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import itertools

from utils.common import read_annotations
from PIL import Image, ImageFile

from .augmentations import get_transform
from .perturbations import get_train_perturbations, get_test_perturbations
from random import random
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, annotations, config, balance=False, isTrain=False, transform=None, perturbation=None):
        self.resize_size = config.resize_size

        self.balance = balance
        self.isTrain = isTrain
        self.config = config
        self.transform = transform
        self.perturbation = perturbation
        paths, labels = zip(*annotations)
        self.class_num = len(set(labels))
        self.data = [[(x, y) for x, y in zip(paths, labels) if y == lab] for lab in set(labels)]
        # select the same number of samples for each class
        if config.samples_per_class is not None and config.samples_per_class > 0:
            self.data = [x[:min(len(x), config.samples_per_class)] for x in self.data]
        if not balance:
            self.data = [list(itertools.chain(*self.data))]

    def __len__(self):
        return max([len(subset) for subset in self.data])

    def __getitem__(self, index):
        if self.balance:
            labs = []
            imgs = []
            img_paths = []
            for i in range(self.class_num):
                safe_idx = index % len(self.data[i])
                img_path, lab = self.data[i][safe_idx]
                img = self.load_sample(img_path)

                labs.append(lab)
                imgs.append(img)
                img_paths.append(img_path)

            return torch.cat([imgs[i].unsqueeze(0) for i in range(self.class_num)]), \
                torch.tensor(labs, dtype=torch.long), img_paths
        else:
            img_path, lab = self.data[0][index]
            img = self.load_sample(img_path)
            lab = torch.tensor(lab, dtype=torch.long)

            return img, lab, img_path

    def load_sample(self, img_path):
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.perturbation:
            img = self.perturbation(img)
        return self.transform(img)

class BaseData(object):
    def __init__(self, train_data_path, val_data_path,
                 test_data_path, out_data_paths,
                 opt, config, transform=None):
        
        train_transform, test_transform = get_transform(config)
        train_perturbations = get_train_perturbations(config)
        test_perturbations = get_test_perturbations(config)

        train_set = ImageDataset(read_annotations(train_data_path, opt.debug), config, balance=True, isTrain=True, transform=train_transform, perturbation=train_perturbations)
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,
            batch_size=config.train_batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )

        val_set = ImageDataset(read_annotations(val_data_path, opt.debug), config, balance=False, transform=test_transform, perturbation=test_perturbations)
        val_loader = DataLoader(
            dataset=val_set,
            num_workers=config.num_workers,
            batch_size=config.eval_batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )

        tsne_set = ImageDataset(read_annotations(test_data_path, opt.debug), config, balance=True, transform=test_transform, perturbation=test_perturbations)
        tsne_loader = DataLoader(
            dataset=tsne_set,
            num_workers=config.num_workers,
            batch_size=config.train_batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

        test_set = ImageDataset(read_annotations(test_data_path, opt.debug), config, balance=False, transform=test_transform, perturbation=test_perturbations)
        test_loader = DataLoader(
            dataset=test_set,
            num_workers=config.num_workers,
            batch_size=config.eval_batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )

        out_loaders = {}
        for name, out_data_path in out_data_paths.items():
            out_set = ImageDataset(read_annotations(out_data_path, opt.debug), config, balance=False, transform=test_transform, perturbation=test_perturbations)
            out_loader = DataLoader(
                dataset=out_set,
                num_workers=config.num_workers,
                batch_size=config.eval_batch_size,
                pin_memory=True,
                shuffle=True,
                drop_last=False
            )
            out_loaders[name] = out_loader

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.out_loaders = out_loaders
        self.tsne_loader = tsne_loader

        print('train: {}, val: {}, test {}'.format(len(train_set), len(val_set), len(test_set)))
        for name, out_loader in out_loaders.items():
            print(f'{name}: {len(out_loader.dataset)}')