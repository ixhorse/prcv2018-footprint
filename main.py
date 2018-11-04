# -*- coding: utf-8 -*-
"""
main
"""

import os, sys
import configparser
import shutil
from train import train
from predict import predict
from aug import dataset_aug

def read_ini(ini_path):
    assert os.path.exists(ini_path)

    config = configparser.ConfigParser()
    config.read(ini_path, encoding="utf-8")
    return config

def data_prepare(config):
    train_path = config.get('MAIN', 'RefPath')
    test_path = config.get('MAIN', 'TestPath')
    task = config.getint('MAIN', 'Task')
    pretrain = config.getboolean('MODEL', 'PreTrain')
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
    assert task in [1, 2]

    # make root dataset dir
    tmp_datadir = './dataset'
    task1_path = os.path.join(tmp_datadir, 'test', 'shibie')
    task2_path = os.path.join(tmp_datadir, 'test', 'wajue')
    if os.path.exists(tmp_datadir):
        # return
        shutil.rmtree(tmp_datadir)
    os.mkdir(tmp_datadir)

    ##
    print('--copy train data.')
    dataset_aug(train_path, os.path.join(tmp_datadir, 'train_aug'))

    ##
    print('--convert train.txt to utf-8.')
    # convert to utf-8
    attr_path = os.path.join(train_path, 'train.txt')
    with open(attr_path, 'r', encoding='gbk') as f:
        info = f.readlines()
    with open('./train.txt', 'w', encoding='utf-8') as f:
        f.writelines(info)

    ##
    print('--copy test data.')
    test_aug_dir = os.path.join(tmp_datadir, 'test_aug')
    os.mkdir(os.path.join(tmp_datadir, 'test_aug'))
    if task == 1:
        os.makedirs(task1_path)
        shutil.copytree(test_path, os.path.join(task1_path, '0'))
        dataset_aug(task1_path, os.path.join(test_aug_dir, 'shibie'))
    else:
        os.makedirs(task2_path)
        shutil.copytree(test_path, os.path.join(task2_path, '0'))
        dataset_aug(task2_path, os.path.join(test_aug_dir, 'wajue'))        

    print('--make output dir.')
    if not os.path.exists('./output'):
        os.mkdir('./output')

if __name__ == '__main__':
    ini_path = './config.ini'
    config = read_ini(ini_path)

    task = config.getint('MAIN', 'Task')
    print('------Task %d------' % task)

    print('preparing data.')
    data_prepare(config)
    pretrain = config.getboolean('MODEL', 'PreTrain')
    validation = config.getboolean('MAIN', 'Validation')
    if pretrain:
        predict(config)
    else:
        model_list = train(config, validation=validation)
        predict(config, model_list)
