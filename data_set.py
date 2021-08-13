#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_set.py
# @Author: yanms
# @Date  : 2021/8/5 16:46
# @Desc  :
import argparse
import random

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


def _data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     划分比例
    :param shuffle:   是否打乱
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


class DataGenerator(object):
    def __init__(self, file_path, ratio):
        super(DataGenerator, self).__init__()
        with open(file_path + 'train.txt', encoding='utf-8') as f:
            datas = f.readlines()
            train_data = []
            validate_data = []
            all_items = []
            train_items = []
            validate_items = []
            for line in datas:
                if len(line) > 0:
                    line = line.strip('\n').split(' ')
                    user = line[0]
                    items = line[1:]
                    all_items.append(list(map(eval, items)))
                    train_item, validate_item = _data_split(items, ratio)
                    train_one = [user]
                    train_one.extend(train_item)
                    train_data.append(list(map(eval, train_one)))
                    validate_one = [user]
                    validate_one.extend(validate_item)
                    validate_data.append(list(map(eval, validate_one)))
                    train_items.append(list(map(eval, train_item)))
                    validate_items.append(list(map(eval, validate_item)))
            self.train_data = train_data
            self.validate_data = validate_data
            self.all_items = all_items
            self.train_items = train_items
            self.validate_items = validate_items
            self.validate_items_length = [len(x) for x in validate_items]
        with open(file_path + 'test.txt', encoding='utf-8') as f:
            test_data = []
            for line in datas:
                if len(line) > 0:
                    line = line.strip('\n').split(' ')
                    test_data.append(list(map(eval, line)))
            self.test_data = test_data


class DataSet(Dataset):
    def __init__(self, generator: DataGenerator, type):
        self.user_count = 52643
        self.item_count = 91599
        self.generator = generator
        self.data = None
        self.type = type
        self.init()

    def __getitem__(self, idx):
        if self.type.lower() == 'train':
            # 生成正负样本对
            user = int(self.data[idx][0])
            items = self.data[idx][1:]
            positive = int(random.sample(items, 1)[0])
            negative = random.randint(0, self.item_count - 1)
            while negative in self.generator.all_items[idx]:
                negative = random.randint(0, self.item_count - 1)
            return np.array([user, positive, negative])
        else:
            user = int(self.data[idx][0])
            return user

    def __len__(self):
        return len(self.data)

    def init(self):
        self.data = self.generator.train_data


if __name__ == '__main__':
    type = 'train'
    file_path = './data/amazon-book/'
    ratio = [0.8, 0.2]
    #
    data_generator = DataGenerator(file_path, 0.8)
    train = DataSet(data_generator, 'validate')
    data_loader = DataLoader(dataset=train, batch_size=3, shuffle=False)
    for index, item in enumerate(data_loader):
        print(index, '-----', item)
