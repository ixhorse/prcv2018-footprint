# -*- coding: utf-8 -*-
"""
split train and test
"""

import shutil
import os, sys
import random

def split(task, src_data_path, train_path, valid_path):
    print('--split train and test set.')
    if os.path.exists(train_path):
        return
        shutil.rmtree(train_path)
    if os.path.exists(valid_path):
        shutil.rmtree(valid_path)
    os.mkdir(train_path)
    os.mkdir(valid_path)

    # split
    if task == 1:
        for idx in os.listdir(src_data_path):
            # make dir for class
            os.mkdir(os.path.join(train_path, idx))
            os.mkdir(os.path.join(valid_path, idx))

            # copy image, last is for valid set
            im_list = os.listdir(os.path.join(src_data_path, idx))
            for img in im_list[:-1]:
                shutil.copy(os.path.join(src_data_path, idx, img),
                            os.path.join(train_path, idx, img))
            shutil.copy(os.path.join(src_data_path, idx, im_list[-1]),
                            os.path.join(valid_path, idx, im_list[-1]))
    else:
        index = list(range(3000))
        random.shuffle(index)

        for i, idx in enumerate(os.listdir(src_data_path)):
            if i in index[:2500]:
                # train set
                dest_path = train_path
            else:
                # val set
                dest_path = valid_path
            os.mkdir(os.path.join(dest_path, idx))
            # copy image
            im_list = os.listdir(os.path.join(src_data_path, idx))
            for img in im_list:
                shutil.copy(os.path.join(src_data_path, idx, img),
                            os.path.join(dest_path, idx, img))