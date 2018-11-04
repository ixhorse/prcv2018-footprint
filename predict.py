# -*- coding: utf-8 -*-
"""
test model
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import os
import pickle
import numpy as np
from pandas import DataFrame
from utils import test_task1, test_task2
from condensenet import CondenseNet_1, CondenseNet_2
from read_attr import FootAttr


def predict(config, model_list=None):
    data_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize([224, 112]),
            transforms.ToTensor(),
            transforms.Normalize([0.054], [0.110])])

    task = config.getint('MAIN', 'Task')
    if task == 1:
        test_data_dir = './dataset/test_aug/shibie'
        CondenseNet = CondenseNet_1
        num_classes = 3000
    else:
        test_data_dir = './dataset/test_aug/wajue'
        CondenseNet = CondenseNet_2
        num_classes = 2
    test_dataset = datasets.ImageFolder(test_data_dir, data_transform)
    testloader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=128,
                                            shuffle=False, 
                                            num_workers=8)
    print('\n--test.')
    print(len(test_dataset))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pretrain = config.getboolean('MODEL', 'PreTrain')
    if pretrain:
        model1 = CondenseNet(stages=(4,6,8,12,10), growth=(8,16,32,64,128), group_1x1=4, group_3x3=4, bottleneck=4, 
                            condense_factor=4, dropout_rate=0., num_classes=num_classes, model_type=1)
        model1 = model1.to(device)
        model2 = CondenseNet(stages=(4,6,8,12,10), growth=(8,16,32,64,128), group_1x1=4, group_3x3=4, bottleneck=4, 
                            condense_factor=4, dropout_rate=0., num_classes=num_classes, model_type=2)
        model2 = model2.to(device)
        model_list = [model1, model2]
        for i, model in enumerate(model_list):
            model_path = config.get('MODEL', 'ModelPath_type%d'%(i+1))
            assert os.path.exists(model_path)
            if torch.cuda.is_available():
                pretrain = torch.load(model_path)
            else:
                pretrain = torch.load(model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(pretrain)

    img_names = sorted(os.listdir(os.path.join(test_data_dir, '0')))

    if task == 1:
        prob1 = test_task1(model_list[0], testloader, device)
        prob2 = test_task1(model_list[1], testloader, device)
        prob = (prob1 + prob2) / 2
        preds = np.argmax(prob, axis=1)
    
        class_names = np.array(sorted(os.listdir('./dataset/train_aug')))
        data = {'filename':img_names, 'preds':class_names[preds]}
        pd = DataFrame(data, columns=['filename', 'preds'])
        pd.to_csv('./output/result_task1.txt', index=False, header=False)
    else:
        gender_prob1, age1, height1, weight1 = test_task2(model_list[0], testloader, device)
        gender_prob2, age2, height2, weight2 = test_task2(model_list[1], testloader, device)
        gender_prob = (gender_prob1 + gender_prob2) / 2
        gender = np.argmax(gender_prob, axis=1)
        age = (age1 + age2) / 2
        height = (height1 + height2) / 2
        weight = (weight1 + weight2) / 2

        attr = FootAttr('./train.txt')
        gender_mapping = {0:'男', 1:'女'}

        gender = [gender_mapping[x] for x in gender]
        age = age * attr.std['age'] + attr.mean['age']
        height = height * attr.std['height'] + attr.mean['height']
        weight = weight * attr.std['weight'] + attr.mean['weight']
        age = np.rint(age).astype(np.int64)
        height = np.rint(height).astype(np.int64)
        weight = np.rint(weight).astype(np.int64)

        data = {'filename':img_names, 'gender':gender, 'age':age, 'height':height, 'weight':weight}
        pd = DataFrame(data, columns=['filename', 'gender', 'age', 'height', 'weight'])
        pd.to_csv('./output/result_task2.txt', index=False, header=False)
