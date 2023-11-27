import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

import data.transforms as data_transforms
from data.randaugment import RandAugment


class ImagenetDataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir
        self.img_path = self.root + '/imagenet/'
        self.web_path = self.root + '/WebVision/'
        self.transform = transform
        self.val_data = []
        with open(os.path.join(self.web_path, 'info/synsets.txt')) as f:
            lines = f.readlines()
        synsets = [x.split()[0] for x in lines]
        for c in range(num_class):
            class_path = os.path.join(self.img_path+'val_img/', synsets[c])
            imgs = os.listdir(class_path)
            for img in imgs:
                self.val_data.append([c, os.path.join(class_path, img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.val_data)


class WebvisionDataset(Dataset):
    def __init__(
            self, root_dir, transform, mode, epoch=None,
            pred=None, probability=None, log=None, num_class=50
    ):
        self.root = root_dir
        self.data_path = '%s/WebVision/' % root_dir
        self.transform = transform
        self.mode = mode

        if self.mode == 'test':
            with open(self.root + '/WebVision/info/val_filelist.txt') as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target
        else:
            with open(self.root +
                      '/WebVision/info/train_filelist_google.txt') as f:
                lines = f.readlines()
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    train_imgs.append(img)
                    self.train_labels[img] = target
            if self.mode == 'all':
                self.train_imgs = train_imgs
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    self.probability = [probability[i] for i in pred_idx]
                    if log != '' and os.path.isfile(log):
                        with open(log, 'a') as f:
                            f.write('{} {:.2f}\n'.format(
                                epoch, float(pred.sum())/len(train_imgs)))
                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(self.data_path + img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.data_path + img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(self.data_path + img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index, img_path
        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(
                self.data_path + 'val_images_256/' + img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)


class WebvisionDataloader():
    def __init__(self, root_dir, batch_size, num_workers,
                 num_class=50, num_batches=None, strong=False):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.strong = strong

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.transform_train = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        if self.strong:
            self.strong_aug = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            self.transform_train = data_transforms.FourCropsTransform(
                self.transform_train, self.transform_train,
                self.strong_aug, self.strong_aug
            )
        self.transform_test = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        self.transform_imagenet = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def run(self, mode, pred=None, prob=None, epoch=None, log=None, path=None):
        if mode == 'warmup':
            all_dataset = WebvisionDataset(
                root_dir=self.root_dir, transform=self.transform_train,
                mode="all", num_class=self.num_class)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader

        elif mode == 'train':
            labeled_dataset = WebvisionDataset(
                root_dir=self.root_dir, transform=self.transform_train,
                mode="labeled", epoch=epoch, num_class=self.num_class,
                pred=pred, probability=prob, log=log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)

            unlabeled_dataset = WebvisionDataset(
                root_dir=self.root_dir, transform=self.transform_train,
                mode="unlabeled", num_class=self.num_class, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = WebvisionDataset(
                root_dir=self.root_dir, transform=self.transform_test,
                mode='test', num_class=self.num_class)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = WebvisionDataset(
                root_dir=self.root_dir, transform=self.transform_test,
                mode='all', num_class=self.num_class)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return eval_loader

        elif mode == 'val':
            imagenet_val = ImagenetDataset(
                root_dir=self.root_dir, transform=self.transform_imagenet,
                num_class=self.num_class)
            imagenet_loader = DataLoader(
                dataset=imagenet_val,
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return imagenet_loader
