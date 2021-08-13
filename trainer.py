#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : trainer.py
# @Author: yanms
# @Date  : 2021/8/12 17:59
# @Desc  :
import argparse
import copy
import random
import time
from tqdm import tqdm

from torch import optim
from torch.utils.data import DataLoader
from logger import Logger
from metrics import metrics_dict

from data_set import DataGenerator, DataSet
from model import LightGCN
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)  # 为CPU设置种子用于生成随机数
torch.cuda.manual_seed(SEED)  # 为GPU设置种子用于生成随机数
torch.cuda.manual_seed_all(SEED)  # 为多个GPU设置种子用于生成随机数

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging = Logger('trainner', level='debug').logger

class Trainer(object):
    def __init__(self, model, data_generater: DataGenerator, args):
        self.model = model.to(device)
        self.data_generater = data_generater
        self.topk = args.topk
        self.metrics = args.metrics
        self.learning_rate = args.lr
        self.weight_decay = args.decay
        self.batch_size = args.batch_size
        self.min_epoch = args.min_epoch
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.optimizer = self.get_optimizer(self.model)
        self.writer = SummaryWriter('./log/lightGCN-' + str(time.time()))
        self.gt_length = data_generater.validate_items_length
        self.validate_item_list = self.generater_validate_items()

    def get_optimizer(self, model):
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def generater_validate_items(self):
        items = self.data_generater.validate_items
        total = [x for j in items for x in j]
        items = torch.Tensor(total).unsqueeze(-1).repeat(1, 20).to(device)
        items = torch.split(items, self.gt_length)
        return items

    def train_model(self):
        train_dataset = DataSet(self.data_generater, 'train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        min_loss = 10  # 用来保存最好的模型
        best_recall, best_ndcg, best_epoch = 0.0, 0.0, 0
        for epoch in range(self.epochs):
            total_loss = 0.0
            self.model.train()
            start_time = time.time()
            train_data_iter = (
                tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"\033[1;35mTrain {epoch:>5}\033[0m"
                )
            )
            batch_no = 0
            for batch_index, batch_data in train_data_iter:
                batch_data = batch_data.to(device)
                self.optimizer.zero_grad()
                loss = self.model.calculate(batch_data)
                loss.backward()
                self.optimizer.step()
                batch_no = batch_index
                total_loss += loss

            total_loss = total_loss / (batch_no + 1)
            # 记录loss到tensorboard可视化
            self.writer.add_scalar('training loss', total_loss, epoch+1)
            epoch_time = time.time() - start_time
            logging.info('epoch %d %.2fs train loss is [%.4f] ' % (epoch + 1, epoch_time, total_loss))

            # evaluate
            metric_dict = self.evaluate(epoch)
            recall, ndcg = metric_dict['recall@' + str(self.topk)], metric_dict['ndcg@' + str(self.topk)]
            if epoch > self.min_epoch and recall > best_recall:
                best_recall, best_ndcg, best_epoch = recall, ndcg, epoch
                best_model = copy.deepcopy(self.model)
                # 保存最好的模型
                self.save_model(best_model)
        logging.info(f"training end, best iteration %d, results: recall@{self.topk}: %s, ndgc@{self.topk}: %s" %
                     (best_epoch+1, best_recall, best_ndcg))

    @torch.no_grad()
    def evaluate(self, epoch):
        validate_dataset = DataSet(self.data_generater, 'validate')
        data_loader = DataLoader(dataset=validate_dataset, batch_size=self.batch_size)
        self.model.eval()
        start_time = time.time()
        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        train_items = self.data_generater.train_items
        for batch_index, batch_data in iter_data:
            batch_data = batch_data.to(device)
            scores = self.model.predict(batch_data)
            # 替换训练集中使用过的item为无穷小
            for user in batch_data:
                items = train_items[user]
                user_score = scores[(user-batch_index*self.batch_size)]
                user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, self.topk, dim=-1)
                gt_items = self.validate_item_list[user]
                mask = (topk_idx - gt_items == 0)
                mask = torch.sum(mask, dim=0) == 1
                topk_list.append(mask.to('cpu'))

        topk_list = torch.cat(topk_list).view(-1, self.topk).numpy()
        metric_dict = self.calculate_result(topk_list, np.array(self.gt_length), epoch)
        epoch_time = time.time() - start_time
        logging.info(f"evaluator %d cost time %.2fs, result: %s " % (epoch, epoch_time, metric_dict.__str__()))
        return metric_dict

    def calculate_result(self, topk_list, gt_len, epoch):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)
        result_list = np.stack(result_list, axis=0).mean(axis=1)
        metric_dict = {}
        for metric, value in zip(self.metrics, result_list):
            key = '{}@{}'.format(metric, self.topk)
            metric_dict[key] = np.round(value[self.topk - 1], 4)
            self.writer.add_scalar('evaluate ' + metric, metric_dict[key], epoch + 1)
        return metric_dict

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.pth')



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--lr', type=int, default=0.001, help='')
    parser.add_argument('--reg_weight', type=int, default=1e-4, help='')
    parser.add_argument('--decay', type=float, default=1e-4, help='')
    parser.add_argument('--negative_count', type=str, default=4, help='')
    parser.add_argument('--batch_size', type=int, default=512, help='')
    parser.add_argument('--epochs', type=str, default=300, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--topk', type=str, default=20, help='')
    parser.add_argument('--metrics', type=str, default=['recall', 'ndcg'], help='')
    parser.add_argument('--model_path', type=str, default='./check_point/', help='')
    parser.add_argument('--ck_path', type=str, default='', help='')
    parser.add_argument('--if_load_model', type=bool, default=False, help='')

    parser.add_argument('--type', type=str, default='train', help='')
    parser.add_argument('--file_path', type=str, default='./data/amazon-book/', help='')
    parser.add_argument('--ratio', type=str, default=[0.8, 0.2], help='')
    parser.add_argument('--user_count', type=int, default=52643, help='')
    parser.add_argument('--item_count', type=int, default=91599, help='')
    parser.add_argument('--n_layers', type=int, default=3, help='')

    args = parser.parse_args()

    args.device = device

    data_generator = DataGenerator(args.file_path, 0.8)
    train_data = DataSet(data_generator, args.type)
    model = LightGCN(args, data_generator.train_data)
    logging.info(model)
    trainer = Trainer(model, data_generator, args)
    trainer.train_model()
    # trainer.evaluate(12)
