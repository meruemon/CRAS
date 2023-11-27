import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

import data.transforms as data_transformas
from data.randaugment import RandAugment


class Clothing1mDataset(Dataset):
    def __init__(self, root, transform, mode, num_samples=0, epoch=None,
                 pred=None, probability=None, log=None, paths=None,
                 num_class=14):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}

        with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                entry = line.split()
                img_path = '%s/images/' % self.root + entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                entry = line.split()
                img_path = '%s/images/' % self.root + entry[0][7:]
                self.test_labels[img_path] = int(entry[1])

        if mode == 'test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    img_path = '%s/images/' % self.root + line[7:]
                    self.test_imgs.append(img_path)
        elif mode == 'val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    img_path = '%s/images/' % self.root + line[7:]
                    self.val_imgs.append(img_path)
        else:
            if mode == 'all':
                img_paths = []
                with open('%s/noisy_train_key_list.txt' % self.root, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        img_path = '%s/images/' % self.root + line[7:]
                        img_paths.append(img_path)
                random.shuffle(img_paths)
                class_num = torch.zeros(num_class)
                self.train_imgs = []
                for impath in img_paths:
                    label = self.train_labels[impath]
                    if class_num[label] < (num_samples / num_class) and \
                            len(self.train_imgs) < num_samples:
                        self.train_imgs.append(impath)
                        class_num[label] += 1
                random.shuffle(self.train_imgs)
            elif mode == "labeled":
                train_imgs = paths
                pred_idx = pred.nonzero()[0]
                self.train_imgs = [train_imgs[i] for i in pred_idx]
                self.probability = [probability[i] for i in pred_idx]
                if log is not None and os.path.isfile(log):
                    with open(log, 'a') as f:
                        f.write('{} {:.2f}\n'.format(
                            epoch, float(len(self.train_imgs))/len(train_imgs)
                            ))
            elif mode == "unlabeled":
                train_imgs = paths
                pred_idx = (1 - pred).nonzero()[0]
                self.train_imgs = [train_imgs[i] for i in pred_idx]
                self.probability = [probability[i] for i in pred_idx]

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index, img_path
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        if self.mode == 'val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)


class Clothing1mDataloader():
    def __init__(self, root, batch_size, num_workers,
                 num_class=14, num_batches=1000, strong=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
        self.num_class = num_class
        self.strong = strong

        mean, std = (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        if self.strong:
            self.strong_aug = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            self.transform_train = data_transformas.FourCropsTransform(
                self.transform_train, self.transform_train,
                self.strong_aug, self.strong_aug
            )

        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def run(self, mode, pred=None, prob=None, epoch=None, log=None, path=None):
        if mode == 'warmup':
            warmup_dataset = Clothing1mDataset(
                self.root, transform=self.transform_train, mode='all',
                num_samples=self.num_batches*self.batch_size*2,
                num_class=self.num_class)
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)
            return warmup_loader

        elif mode == 'train':
            labeled_dataset = Clothing1mDataset(
                self.root, transform=self.transform_train, mode='labeled',
                pred=pred, probability=prob, log=log, paths=path,
                epoch=epoch, num_class=self.num_class)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = Clothing1mDataset(
                self.root, transform=self.transform_train, mode='unlabeled',
                pred=pred, probability=prob, paths=path,
                num_class=self.num_class)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_loader, unlabeled_loader

        elif mode == 'eval_train':
            eval_dataset = Clothing1mDataset(
                self.root, transform=self.transform_test, mode='all',
                num_samples=self.num_batches*self.batch_size,
                num_class=self.num_class)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size*32,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

        elif mode == 'test':
            test_dataset = Clothing1mDataset(
                self.root, transform=self.transform_test, mode='test',
                num_class=self.num_class)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size*32,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'val':
            val_dataset = Clothing1mDataset(
                self.root, transform=self.transform_test, mode='val',
                num_class=self.num_class)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size*32,
                shuffle=False,
                num_workers=self.num_workers)
            return val_loader
