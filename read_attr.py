# -*- coding: utf-8 -*-
"""
处理属性
"""

import os
import numpy as np
import pandas as pd
import re
import pickle

class FootAttr():
    def __init__(self, attr_path):
        assert os.path.exists(attr_path)
        self.dest_path = './attr.txt'

        if not os.path.exists(self.dest_path):
            fattr = []
            with open(attr_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = re.split(r'[ ]+', line)
                    line = ','.join(line)
                    fattr.append(line)
            with open(self.dest_path, 'w', encoding='utf-8') as f:
                f.writelines(fattr)

        self.attr = self.read()
        self.mean = self.attr.mean()
        self.std = self.attr.std()

    def read(self):
        attr = pd.read_csv(self.dest_path, sep=',')
        attr.drop(['样本数', '([]处代表没有信息)'], axis=1, inplace=True)
        attr.rename(columns={'性别':'gender', '年龄':'age', '身高':'height', '体重':'weight'}, inplace = True)

        gender_mapping = {'男':0, '女':1}
        attr['gender'] = attr['gender'].map(gender_mapping)

        # attr[['height', 'weight']] = attr[['height', 'weight']].replace('[]', np.nan)

        for col in ['age', 'height', 'weight']:
            attr[col] =  pd.to_numeric(attr[col], errors='coerce')
        
        # fill na with mean
        mean_M = attr.loc[attr['gender'] == 0].mean()
        mean_F = attr.loc[attr['gender'] == 1].mean()
        n_row, n_col = attr.shape
        for i in range(n_row):
            for j in range(2, n_col):
                if np.isnan(attr.ix[i, j]):
                    attr.ix[i, j] = mean_M.ix[j] if attr.ix[i, 0] == 0 else mean_F.ix[j]

        attr.fillna(method='bfill', inplace=True)
        return attr

    def get_gender(self):
        return self.attr['gender'].values

    def get_age(self):
        return (self.attr['age'].values - self.mean['age']) / self.std['age']

    def get_height(self):
        return (self.attr['height'].values - self.mean['height']) / self.std['height']

    def get_weight(self):
        return (self.attr['weight'].values - self.mean['weight']) / self.std['weight']

if __name__ == '__main__':
    attr = FootAttr('./train.txt')
    print(attr.attr.isnull().any())
