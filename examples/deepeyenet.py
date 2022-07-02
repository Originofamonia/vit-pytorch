from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile
import json
import argparse, yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from vit_pytorch.efficient import ViT

# os.environ['CUDA_VISIBLE_DEVICES'] = 0

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class DeepEyeNetDataset(Dataset):
    def __init__(self, args, transform=None):
        self.datapath = args.datapath
        self.split_file = args.split  # train, valid, or test
        with open(f'{args.split}', 'r') as f3:
            self.json_data = json.load(f3)
        self.transform = transform
        with open(os.path.join(args.datapath, 'least8_label2idx.json'), 'r') as f1:
            self.label2idx = json.load(f1)
        with open(os.path.join(args.datapath, 'least8_idx2label.json'), 'r') as f2:
            self.idx2label = json.load(f2)
        self.labelsize = len(self.label2idx)

    def __len__(self):
        self.filelength = len(self.json_data)
        return self.filelength

    def __getitem__(self, idx):
        example = self.json_data[idx]
        for k, v in example.items():
            img = Image.open(os.path.join(self.datapath, k))
            img_transformed = self.transform(img)
            if img_transformed.shape[0] > 3:  
                img_transformed = img_transformed[:3]
            keywords = v['keywords'].split(', ')
            label_indices = []
            for word in keywords:
                if word in self.label2idx:
                    label_indices.append(self.label2idx[word])
            # label_indices = [self.label2idx[word] for word in keywords]
            label = np.zeros(self.labelsize)
            label[label_indices] = 1
            label = np.float32(label)
            return img_transformed, label


def get_dataset_split(args):
    train_split = os.path.join(args.datapath, 'filtered_DeepEyeNet_train.json')
    val_split = os.path.join(args.datapath, 'filtered_DeepEyeNet_valid.json')
    test_split = os.path.join(args.datapath, 'filtered_DeepEyeNet_test.json')
    return train_split, val_split, test_split


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='MCAN Args')

    parser.add_argument('--run', dest='run_mode',
                        choices=['train', 'val', 'test', 'visualize'],
                        type=str, default='train')

    parser.add_argument('--model', dest='model',
                      choices=['small', 'large'],
                      default='large', type=str)

    parser.add_argument('--datapath', type=str,
                        default='/drive/qiyuan/retinal/deepeyenet/')

    parser.add_argument('--split', dest='train_split', type=str,
                      choices=['train', 'train+val', 'train+val+vg'],
                      help="set training split, "
                           "eg.'train', 'train+val+vg'"
                           "set 'train' can trigger the "
                           "eval after every epoch",)

    parser.add_argument('--eval_every_epoch', default=True, type=bool,
                      help='set True to evaluate the '
                           'val split when an epoch finished'
                           "(only work when train with "
                           "'train' split)",)

    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--max_epoch', default=30, type=int)
    parser.add_argument('--lr_base', default=3e-5, type=float)
    parser.add_argument('--gamma', default=0.7, type=float)
    parser.add_argument('--seed', default=444, type=int)
    parser.add_argument('--gpu', default='1', type=str,
                        help="gpu select, eg.'0, 1, 2'",)
    args = parser.parse_args()
    return args


def train(args, model, train_loader, valid_loader, test_loader):

    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model, device_ids=args.devices)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    # BCELoss's preds must be within [0, 1]
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_base)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(args.max_epoch):
        epoch_acc, epoch_loss = 0, 0
        targets, preds = [], []
        pbar = tqdm(train_loader)
        model.train()
        for i, (data, label) in enumerate(pbar):
            data = data.cuda()
            label = label.cuda()
            
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds.append(output.detach().cpu().numpy())
            targets.append(label.detach().cpu().numpy())
            epoch_loss += loss / len(train_loader)
            # acc = (output.argmax(dim=1) == label).float().mean()
            # epoch_acc += acc / len(train_loader)
            pbar.set_description(f'e: {epoch}, loss: {loss.item():.3f}')
        targets = np.vstack(targets)
        preds = np.vstack(preds)
        target_sums = targets.sum(axis=0)
        zero_classes = target_sums == 0
        assert sum(zero_classes) == 0

        perclass_roc = roc_auc_score(targets, preds, average=None)
        micro_roc = roc_auc_score(targets, preds, average='micro')
        macro_roc = roc_auc_score(targets, preds, average='macro')
        # print(f'e: {epoch}, loss: {epoch_loss:.3f}, train acc: {epoch_acc:.3f}, \n')
        print(f'e: {epoch}, loss: {epoch_loss:.3f}, train perclass_roc: {perclass_roc},' +
            f' train micro_roc: {micro_roc:.3f}, train macro_roc: {macro_roc:.3f}\n')

        eval(args, model, valid_loader, zero_classes, criterion, split='val')
        eval(args, model, test_loader, zero_classes, criterion, split='test')


def eval(args, model, valid_loader, train_zero_classes, criterion, split='val'):
    model.eval()
    with torch.no_grad():
        epoch_loss, epoch_val_acc = 0, 0
        targets, preds = [], []
        pbar = tqdm(valid_loader)
        for i, (data, label) in enumerate(pbar):
            data = data.cuda()
            label = label.cuda()

            val_output = model(data)
            val_loss = criterion(val_output, label)
            preds.append(val_output.detach().cpu().numpy())
            targets.append(label.detach().cpu().numpy())
            # acc = (val_output.argmax(dim=1) == label).float().mean()
            # epoch_val_acc += acc / len(valid_loader)
            epoch_loss += val_loss / len(valid_loader)

        targets = np.vstack(targets)
        preds = np.vstack(preds)
        target_sums = targets.sum(axis=0)
        zero_classes = target_sums == 0
        assert sum(zero_classes) == 0
        # filtered_targets = np.delete(targets, zero_classes, axis=1)
        # filtered_preds = np.delete(preds, zero_classes, axis=1)
        perclass_roc = roc_auc_score(targets, preds, average=None)
        micro_roc = roc_auc_score(targets, preds, average='micro')
        macro_roc = roc_auc_score(targets, preds, average='macro')
        # print(f'{split} loss: {epoch_loss:.3f}, acc: {epoch_val_acc:.3f}\n')
        print(f'{split} loss: {epoch_loss:.3f}, perclass_roc: {perclass_roc}, ' +
            f'micro_roc: {micro_roc:.3f}, macro_roc: {macro_roc:.3f}\n')


def proc(args):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.n_gpu = len(args.gpu.split(','))
    args.devices = [_ for _ in range(args.n_gpu)]
    return args

def main():
    args = parse_args()
    args = proc(args)
    seed_everything(args.seed)

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    train_split, val_split, test_split = get_dataset_split(args)
    args.split = train_split
    train_data = DeepEyeNetDataset(args, train_transforms)
    args.split = val_split
    val_data = DeepEyeNetDataset(args, val_transforms)
    args.split = test_split
    test_data = DeepEyeNetDataset(args, val_transforms)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    efficient_transformer = Linformer(
        dim=128,
        seq_len=49+1,  # 7x7 patches + 1 cls-token
        depth=12,
        heads=8,
        k=64
    )

    model = ViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=train_data.labelsize,
        transformer=efficient_transformer,
        channels=3,
    ).cuda()
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # model.fc = nn.Linear(512, train_data.labelsize)
    # model.cuda()
    train(args, model, train_loader, valid_loader, test_loader)


if __name__ == '__main__':
    main()
