# -*- coding: utf-8 -*-
"""
2018/8/26
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image

def train_task1(model, criterion, optimizer, scheduler, num_epochs,
                dataloaders, dataset_sizes, device, validation, attr=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    phases = ['train', 'val'] if validation else ['train']

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # acc
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if validation:
        model.load_state_dict(best_model_wts)
    return model

def train_task2(model, criterion, optimizer, scheduler, num_epochs,
                dataloaders, dataset_sizes, device, validation, attr=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    phases = ['train', 'val'] if validation else ['train']

    assert attr
    gender = torch.from_numpy(attr.get_gender())
    age = torch.from_numpy(attr.get_age())
    height = torch.from_numpy(attr.get_height())
    weight = torch.from_numpy(attr.get_weight())
    if validation:
        order = {x: np.array(list(map(int, sorted(os.listdir('./dataset/%s_split'%x))))) - 1
                for x in ['train', 'val']}
    else:
        order = {x: np.array(list(map(int, sorted(os.listdir('./dataset/%s_aug'%x))))) - 1
                for x in ['train']}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels_gender = gender[order[phase][labels.data]].to(device)
                labels_age = age[order[phase][labels.data]].view(-1, 1).float().to(device)
                labels_height = height[order[phase][labels.data]].view(-1, 1).float().to(device)
                labels_weight = weight[order[phase][labels.data]].view(-1, 1).float().to(device)
                # print(labels_age.cpu().numpy().flatten())
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    out1, out2, out3, out4 = model(inputs)
                    # print(out1, out2, out3, out4)
                    _, preds = torch.max(out1, 1)
                    loss1 = criterion(out1, labels_gender)
                    loss2 = nn.SmoothL1Loss()(out2, labels_age)
                    loss3 = nn.SmoothL1Loss()(out3, labels_height)
                    loss4 = nn.SmoothL1Loss()(out4, labels_weight)
                    # print(loss1, loss2, loss3, loss4)
                    loss = loss1 + loss2 + loss3 + loss4
                    # if np.isnan(loss.detach().cpu().numpy()):
                    #     print(labels_age.cpu().numpy().flatten())
                    #     print(labels_gender.cpu().numpy().flatten())
                    #     print(labels_height.cpu().numpy().flatten())
                    #     print(labels_weight.cpu().numpy().flatten())
                            # + nn.SmoothL1Loss()(out2[:, 2].view(-1, 1), labels_weight)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels_gender.data)

            # acc
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc_gender: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if validation:
        model.load_state_dict(best_model_wts)
    return model


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.054])
    std = np.array([0.110])
    inp = std * inp + mean
    print(inp.shape)
    inp = np.clip(inp, 0, 1)
    inp = inp[:,:,0]
    plt.imshow(inp, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated

def save_model(param, name):
    torch.save(param, './output/'+name+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M') + '.pth')

def test_task1(model, dataloader, device):
    since = time.time()
    model.eval()   # Set model to evaluate mode
    all_pred = []

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            probability = nn.Softmax(dim=1)(outputs).cpu().numpy()
            # _, preds = torch.max(outputs, 1)

        all_pred.append(probability)

    all_pred = np.vstack(all_pred)

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return all_pred

def test_task2(model, dataloader, device):
    since = time.time()
    model.eval()   # Set model to evaluate mode
    gender_result = []
    age_result = []
    height_result = []
    weight_result = []

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            out1, out2, out3, out4 = model(inputs)
            out1 = nn.Softmax(dim=1)(out1)

        gender_result.append(out1.cpu().numpy())
        age_result.append(out2.cpu().numpy().flatten())
        height_result.append(out3.cpu().numpy().flatten())
        weight_result.append(out4.cpu().numpy().flatten())

    gender_result = np.vstack(gender_result)
    age_result = np.hstack(age_result)
    height_result = np.hstack(height_result)
    weight_result = np.hstack(weight_result)

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return gender_result, age_result, height_result, weight_result

# # # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))

# # # Make a grid from batch
# # out = torchvision.utils.make_grid(inputs)
# out = inputs[0]
# print(out.shape)

# imshow(out, title=[class_names[x] for x in classes])