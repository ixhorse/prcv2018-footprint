# -*- coding: utf-8 -*-
"""
2018/7/17
train model
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os, sys
import pickle
from utils import train_task1, train_task2, save_model, imshow
from split import split
from condensenet import CondenseNet_1, CondenseNet_2
from read_attr import *


def train(config, validation=False):

    # params
    task = config.getint('MAIN', 'Task')
    if task == 1:
        num_classes=3000
        num_epochs = 160
        step_size = [100, 150]
        base_lr = 0.001
        train_model = train_task1
        criterion = nn.CrossEntropyLoss()
        CondenseNet = CondenseNet_1
    elif task == 2:
        num_classes=2
        num_epochs = 100
        step_size = [50, 90]
        base_lr = 0.001
        train_model = train_task2
        criterion = nn.CrossEntropyLoss()
        CondenseNet = CondenseNet_2

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize([224, 112]),#[heiht, width]
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.054], [0.110]),
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize([224, 112]),
            transforms.ToTensor(),
            transforms.Normalize([0.054], [0.110]),
        ]),
    }

    data_dir = './dataset'
    assert os.path.exists(data_dir)
    if validation:
        phases = ['train', 'val']
        split(task, './dataset/train_aug', './dataset/train_split', './dataset/val_split')
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x+'_split'),
                                                  data_transforms[x])
                                                  for x in phases}
    else:
        phases = ['train']
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x+'_aug'),
                                                  data_transforms[x])
                                                  for x in phases}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                    batch_size=128,
                                                    shuffle=True, 
                                                    num_workers=8)
                                                    for x in phases}
    dataset_sizes = {x: len(image_datasets[x]) for x in phases}
    class_names = image_datasets['train'].classes
    print(dataset_sizes, len(class_names))

    # attr
    attr = FootAttr('./train.txt') if task > 1 else None

    model_list = []
    for model_type in [1, 2]:
        # model
        model = CondenseNet(stages=(4,6,8,12,10), growth=(8,16,32,64,128), group_1x1=4, group_3x3=4, bottleneck=4, 
                            condense_factor=4, dropout_rate=0., num_classes=num_classes, model_type=model_type)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Observe that all parameters are being optimized
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, weight_decay=0.001)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=step_size, gamma=0.1)
        model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs, 
                            dataloaders, dataset_sizes, device, validation, attr)

        # save params
        save_model(model.state_dict(), 'task%d_type%d' % (task, model_type))
        model_list.append(model)

    return model_list