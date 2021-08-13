#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: yanms
# @Date  : 2021/8/12 10:18
# @Desc  :

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score)).mean()
        return loss


class EMBLoss(nn.Module):
    def __init__(self, norm=2):
        super(EMBLoss, self).__init__()
        self.norm = norm

    def forward(self, weights):
        loss = torch.norm(weights, p=self.norm)
        loss = loss / weights.shape[0]
        return loss


class LightGCN(nn.Module):

    def __init__(self, args, data_list):
        super(LightGCN, self).__init__()
        self.device = args.device
        self.embedding_size = args.embedding_size
        self.user_count = args.user_count
        self.item_count = args.item_count
        self.n_layers = args.n_layers
        self.reg_weight = args.reg_weight
        self.user_embedding = nn.Embedding(self.user_count, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_count, self.embedding_size)
        self.data_list = data_list
        self.interaction_matrix = self.get_interaction_matrix()
        self.A_adj_matrix = self.get_a_adj_matrix()
        self.BPRLoss = BPRLoss()
        self.EMBLoss = EMBLoss()
        self.restore_user_e = None
        self.restore_item_e = None
        self.if_load_model = args.if_load_model
        self.ck_path = args.ck_path
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if self.if_load_model:
            parameters = torch.load(self.ck_path)
            self.load_state_dict(parameters)
        else:
            if isinstance(module, nn.Embedding):
                torch.nn.init.xavier_normal_(module.weight.data)

    def get_a_adj_matrix(self):
        """
        得到 系数矩阵A~
        :return:
        """
        A = sp.dok_matrix((self.user_count + self.item_count, self.user_count + self.item_count), dtype=np.float)
        inter_matrix = self.interaction_matrix
        inter_matrix_t = self.interaction_matrix.T
        data_dict = dict(zip(zip(inter_matrix.row, inter_matrix.col), [1] * inter_matrix.nnz))
        data_dict.update(dict(zip(zip(inter_matrix_t.row, inter_matrix_t.col), [1] * inter_matrix_t.nnz)))
        A._update(data_dict)
        sum_list = (A > 0).sum(axis=1)
        diag = np.array(sum_list.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        A_adj = D * A * D
        A_adj = sp.coo_matrix(A_adj)
        row = A_adj.row
        col = A_adj.col
        index = torch.LongTensor([row, col])
        data = torch.FloatTensor(A_adj.data)
        A_sparse = torch.sparse.FloatTensor(index, data, torch.Size(A_adj.shape))
        return A_sparse.to(self.device)

    def get_interaction_matrix(self):
        inter_list = []
        for line in self.data_list:
            user = line[0]
            items = line[1:]
            for item in items:
                inter_list.append([int(user), int(item)])
        inter_list = np.array(inter_list)
        user_id = inter_list[:, 0]
        item_id = inter_list[:, 1]
        data = np.ones(len(inter_list))
        return sp.coo_matrix((data, (user_id, item_id)), shape=(self.user_count, self.item_count))

    def forward(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        embedding_list = [all_embeddings]
        for i in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.A_adj_matrix, all_embeddings)
            embedding_list.append(all_embeddings)

        total_E = torch.stack(embedding_list, dim=0)
        total_E = torch.mean(total_E, dim=0)
        user_all_embedding, item_all_embedding = torch.split(total_E, [self.user_count, self.item_count])
        return user_all_embedding, item_all_embedding

    def calculate(self, epoch_data):

        user_all_embedding, item_all_embedding = self.forward()
        epoch_data = epoch_data.to(self.device)
        users = epoch_data[:, 0]
        positive = epoch_data[:, 1]
        negative = epoch_data[:, 2]
        users_embedding = user_all_embedding[users.long()]
        positive_embedding = item_all_embedding[positive.long()]
        negative_embedding = item_all_embedding[negative.long()]

        p_scores = torch.mul(users_embedding, positive_embedding)
        n_scores = torch.mul(users_embedding, negative_embedding)
        bpr_loss = self.BPRLoss(p_scores, n_scores)
        reg_loss = self.EMBLoss(users_embedding) + self.EMBLoss(positive_embedding) + self.EMBLoss(negative_embedding)

        loss = bpr_loss + self.reg_weight * reg_loss
        return loss

    def predict(self, epoch_data):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        users = epoch_data.to(self.device)
        user_embedding = self.restore_user_e[users]
        scores = torch.matmul(user_embedding, self.restore_item_e.T)
        return scores
