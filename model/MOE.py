import collections
import os
import math
import pickle
from time import time
from time import strftime
from scipy.sparse.csr import csr_matrix

import numpy as np
# from numpy.random import randint, binomial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from concurrent import futures
from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
# from utils import Logger, set_random_seed
from sklearn.cluster import KMeans
from collections import OrderedDict
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
import warnings
from tqdm import tqdm
import copy
import torch.distributed as dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import KernelPCA
from experiment import EarlyStop
import matplotlib.pyplot as plt
# from softadapt import SoftAdapt

warnings.filterwarnings("ignore")
# PCA sklearn
class MOE(BaseRecommender):
    # add entropy added to the loss (ce), smaller is better
    def __init__(self, dataset, model_conf, device):
        super(MOE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        # config
        self.num_experts = model_conf['num_experts']
        self.num_features = self.num_items
        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']

        self.lr = model_conf['lr']
        self.dropout = model_conf['dropout']
        ### change weight decay
        self.reg = model_conf['reg'] / self.num_experts
        # self.reg = model_conf['reg'] / (self.num_experts + 10000)
        print("weight decay:", self.reg)
        self.anneal = 0.
        self.anneal_cap = model_conf['anneal_cap']
        self.total_anneal_steps = model_conf['total_anneal_steps']

        print("... Model - MOE")
        self.num_local_threads = model_conf['num_local_threads']
        self.update_count = 0
        self.device = device
        # self.es = EarlyStop(15, 'mean')
        # self.beta = model_conf['beta']
        """ Initialize gate weights (number of input features * number of experts * number of tasks)"""

        # # feed forward neural network
        # self.gate_kernels = gate_network(self.num_items, self.num_experts)

        # fixed gate parameters here
        # self.tau = model_conf['tau']
        # self.tau_loss = model_conf['tau_loss']

        # self.gate_kernels.requires_grad_(False)

        # self.gate_kernels = torch.rand(
        #         (self.num_features, self.num_experts)
        #     ).float()
        #
        # self.gate_kernels = nn.Parameter(self.gate_kernels, requires_grad=True)
        #
        # self.gate_bias = nn.Parameter(
        #     torch.zeros(1, self.num_experts), requires_grad=True
        # )

        ## need to change the AE model
        # with open(os.path.join('gate_AE', 'AE_gate.npy'), 'rb') as f:
        #     self.gate_kernels = torch.FloatTensor(np.load(f)).to(self.device)

        # enlft gate init
        # with open(os.path.join(self.dataset.data_dir, self.dataset.data_name, 'mainstream_scores', str(self.num_experts) + '_gate_initialize_enlft_new_preprocess.npy'), 'rb') as f:
        #     # TensorDataset(torch.FloatTensor(np.load(path))), two devices issues, cpu here need to change to gpu
        #     # self.gate_kernels = torch.FloatTensor(np.load(f)).to(self.device)
        #     init = np.load(f)
        #     # self.gate_kernels = torch.FloatTensor(init / np.sum(init, axis=1, keepdims=True)).to(self.device)
        #     self.gate_kernels = F.softmax(torch.FloatTensor(init).to(self.device), dim=1)

        # load train_set data
        # self.train_set = defaultdict(list)
        # for i in range(len(dataset.train_df)):
        #     for j in range(len(dataset.train_df[0])):
        #         if dataset.train_df[i][j] == 1:
        #             self.train_set[i].append(j)

        # load validation data
        self.vali_data = dataset.vali_dict
        self.vali_ = np.zeros((self.num_users, self.num_items))
        for u in self.vali_data:
            self.vali_[u][self.vali_data[u]] = 1

        # self.p_u = (self.num_items - dataset.train_df.sum(1) - self.vali_.sum(1)) / self.num_items
        # self.temp_set = {}
        # for idx in range(self.num_users):
        #     p = self.p_u[idx]
        #     n_idx = np.random.choice(self.num_items, int(100 / p), replace=False).tolist()
        #     # n_idx = torch.randperm(self.num_items)[:int(100 / p)]
        #     self.temp_set[idx] = list(set(n_idx).difference(set(dataset.train_df[idx]).difference(set(self.vali_[idx]))))
        # self.vali_set = copy.deepcopy(dataset.vali_dict)
        # self.vali_set.update(self.temp_set)


        # scaling = StandardScaler()
        # Scaled_data = scaling.fit_transform(data)

        # Set the n_components=3
        # principal = KernelPCA(n_components=self.num_experts, kernel='rbf')
        # principal = SparsePCA(n_components=self.num_experts)
        # data = self.dataset.train_matrix.tocoo().toarray()
        #
        # principal = PCA(n_components=self.num_experts)
        # gate_pca = principal.fit_transform(data)
        # # norm_gate = gate_pca / gate_pca.std(0) # method 1
        # norm_gate = (gate_pca + gate_pca.std(0)) / (gate_pca.std(0) * 2) # method 2

        # self.gate_kernels = F.softmax(torch.FloatTensor(norm_gate / self.tau).to(self.,device)
        # self.gate_kernels = F.softmax(torch.FloatTensor(gate_pca).to(self.device) / 1)

        # self.gate_loss_kernels = F.softmax(torch.FloatTensor(gate_pca / 1).to(self.device))

        self.expert_kernels = nn.ModuleList()
        # self.expert_kernels = nn.ModuleList(
        #     [
        #         MultVAE_Expert(
        #             dataset, model_conf, device
        #         )
        #         # copy.deepcopy(expert)
        #         for i in range(self.num_experts)
        #     ]
        # )
        for i in range(self.num_experts):
            expert = MultVAE_Expert(dataset, model_conf, device)
            # expert.load_state_dict(torch.load(os.path.join('gate_local', str(i) + 'th_gate.p'), map_location=device))
            # # # expert.load_state_dict(torch.load(os.path.join('saves', 'MultVAE', '1_20221208-1805', 'best_model.p'), map_location=device))
            # local_model = copy.deepcopy(expert)
            # noise_encoder = (torch.randn(local_model.encoder[0].weight.size()) * 0.7 + 0).to(device)
            # noise_decoder = (torch.randn(local_model.decoder[0].weight.size()) * 0.7 + 0).to(device)
            # self.add_noise(local_model.encoder[0].weight, noise_encoder)
            # self.add_noise(local_model.decoder[0].weight, noise_decoder)
            # self.expert_kernels.append(local_model)
            self.expert_kernels.append(expert)

        """ Initialize gate bias (number of experts * number of tasks)"""
        # self.dropout = nn.Dropout(0.2)
        # self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        self.to(self.device) ###
        # self.cuda()

        # # calculate weights for weighted loss
        # similarity_dir = os.path.join(dataset.data_dir, dataset.data_name, 'mainstream_scores')
        # similarity_file = os.path.join(similarity_dir, 'MS_similarity.npy')
        # self.ms = np.load(similarity_file)
        # self.ms_ = -np.load(similarity_file)
        # self.weight = self.ms_ - np.min(self.ms_)
        # self.weight = (self.weight / np.max(self.weight)) ** 1 ## 1.4, 1.2,    1.1, 1, 0.9
        # self.weight = self.weight / np.mean(self.weight)

        # # softmax 1
        # self.weight = 1 / self.ms
        # # 10 - 100
        # self.weight = F.softmax(torch.FloatTensor(self.weight) / 55)

        # # scale mainstream users
        # self.weight = 1 / self.ms
        # k = int(self.num_users * 0.2)
        # indices = np.argpartition(self.weight, k)
        # self.weight[indices[:k]] *= 0.7
        # self.weight = F.softmax(torch.FloatTensor(self.weight) / 55)

        # # wl with alpha method
        # self.weight = torch.FloatTensor(1 / self.ms)
        # beta = torch.zeros(self.weight.reshape(-1, 1).shape)
        # alpha = 30 # 10 - 100, 100: 0.9 - 3, 500: 0.98, 1.4
        # self.weight = torch.max(((self.weight - self.weight.mean() + 2 * alpha) / (2 * alpha)).reshape(-1, 1), beta)


        # # softmax 2
        # # 0.01 - 0.0009
        # self.weight = F.softmax(torch.FloatTensor(self.ms) / 0.002)
        # self.weight = (1 - self.weight) / self.num_users

        # # mainstream score dis
        # self.m_dis = F.softmax(torch.FloatTensor(self.ms) / 0.02)

        # user_sort_idx = np.argsort(self.ms)
        # self.test_gate_ids = user_sort_idx
        # self.flag = False
        self.time = strftime('%Y%m%d-%H%M')
        self.vali_batch = None
        self.vali_batch_cuda = None
        # self.alpha = 100000

    # def add_noise(self, weights, noise):
    #     with torch.no_grad():
    #         weights.add_(noise)

    def forward(self, x, batch_idx):
        n = x.shape[0]
        expert_output_list = []
        kl_loss_list = []
        dimension = 0
        ### for checking
        vali_acc = None
        loss = None
        dis_pre = None
        # loss = []

        self.vali_batch = torch.FloatTensor(self.vali_).to(self.device)[batch_idx] ###
        # self.vali_batch = torch.FloatTensor(self.vali_).cuda()[batch_idx]
        mask = 1 - x

        # self.vali_batch = self.vali_[batch_idx]
        # self.vali_batch_cuda = torch.FloatTensor(self.vali_).to(self.device)[batch_idx]
        # mask = (1 - x).detach().cpu().numpy()
        # vali_batch_set = {key: self.vali_data[key] for key in batch_idx}
        # p_u = (self.num_items - x.sum(1) - vali_batch.sum(1)) / self.num_items
        # temp_set = {}
        # for idx, id in enumerate(batch_idx):
        #     p = p_u[idx]
        #     # n_idx = np.random.choice(self.num_items, 100 / p, replace=False).tolist()
        #     n_idx = torch.randperm(self.num_items)[:int(100 / p)]
        #     temp_set[id] = list(set(n_idx).difference(set(x[idx]).difference(set(vali_batch[idx]))))

        # temp_vali_set = {key: self.vali_set[key] for key in batch_idx}
        # vali_loss_list = []
        for i in range(self.num_experts):
            if self.training:
                expert_kernel, kl_loss = self.expert_kernels[i](x)
                kl_loss_list.append(kl_loss)
            else:
                expert_kernel = self.expert_kernels[i](x)
            # weighted here
            vali_loss = -torch.log(torch.exp(expert_kernel) / torch.exp(expert_kernel * mask).sum(1).reshape(-1, 1))
            # vali_loss = -np.log(np.exp(expert_kernel.detach().cpu().numpy()) / np.sum(np.exp(expert_kernel.detach().cpu().numpy() * mask), axis=1).reshape((-1, 1)))

            # for expert dropout distribution
            # vali_loss_dis = (vali_loss * self.vali_batch).sum(1).reshape(-1, 1)

            expert_vali_loss = 1 / (vali_loss * self.vali_batch).sum(1).reshape(-1, 1)
            # expert_vali_loss = 1 / np.sum(vali_loss * self.vali_batch, axis=1).reshape(-1, 1)

            # # expert_loss = 1 / (-(F.log_softmax(expert_kernel, 1) * x).sum(1).reshape(-1, 1))
            #
            # rec = self.expert_kernels[i].predict(np.arange(self.dataset.num_users).tolist(), self.dataset.train_matrix)
            # vali = np.reshape(self.evaluator_vali.evaluate_vali_batch(rec, temp_vali_set, mean=False)['NDCG@20'], (-1, 1))

            # vali_loss_list.append(vali_loss_dis)
            if i == 0:
                # loss = expert_loss
                # vali_acc = vali[batch_idx]
                loss = expert_vali_loss
                # loss = expert_vali_loss
                # vali_acc = vali
                # dis_pre = vali_loss_dis
            else:
                # loss = torch.cat((loss, expert_loss), 1)
                # vali_acc = np.append(vali_acc, vali[batch_idx], axis=1)
                loss = torch.cat((loss, expert_vali_loss), 1)
                # loss = np.append(loss, expert_vali_loss,axis=1)
                # vali_acc = np.append(vali_acc, vali, axis=1)
                # dis_pre = torch.cat((dis_pre, vali_loss_dis), 1)
            aux = expert_kernel
            dimension = expert_kernel.shape[1]
            aux = torch.reshape(aux, (n, dimension))
            expert_output_list.append(torch.reshape(aux, [-1, 1, dimension]))  # [b, 1, dimension]

            # aux = expert_kernel.detach().cpu().numpy()
            # dimension = expert_kernel.shape[1]
            # aux = np.reshape(aux, (n, dimension))
            # expert_output_list.append(np.reshape(aux, (-1, 1, dimension)))
        expert_outputs = torch.cat(expert_output_list, 1)  # size: batch_size X expert X self.dimension
        # expert_outputs = np.concatenate(expert_output_list, axis=1)
        # expert_outputs = torch.FloatTensor(expert_outputs).to(self.device)
        # dis_pre = torch.cat(vali_loss_list, 1)

        # vali_acc = vali_acc / np.sum(vali_acc, axis=1, keepdims=True)
        # loss = F.softmax(loss)
        # loss = loss / loss.sum(1).reshape(-1, 1)
        """ Calculating the gates"""
        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper T x n x E
        # gate_outputs = torch.mm(x, self.gate_kernels).reshape(n, self.num_experts)
        # gate_outputs = gate_outputs.add(self.gate_bias)
        # self.gate = F.softmax(gate_outputs)

        # self.gate = self.gate_kernels(x)

        ## gate initialized
        # self.gate = self.gate_kernels[batch_idx]

        # gate_loss = self.gate_loss_kernels[batch_idx]

        ##
        # entropy = self.gate * torch.log(self.gate)
        # entropy_ig = entropy.mean(1)
        # entropy_ig_norm = entropy_ig / entropy_ig.min()
        # entropy_ig_norm = (1 / entropy_ig_norm)
        ## method 1, 0 for row (exp, sum of one column), 1 for column
        # assign gates dynamically, can use more than one time for gates, or not use some gates, more even is better
        # gate_mean = self.gate.mean(0)
        # entropy_norm = (gate_mean * torch.log(gate_mean)).sum()

        # ## method 2
        # entropy_list = (self.gate * torch.log(self.gate)).sum(1)
        # entropy_norm = entropy_list.sum()

        # method 3, minus beta, 1e-20
        # each user uses different gate, more even is bad
        # gate_scale = self.gate / self.gate.sum(0) # softmax try instead
        # entropy_norm = (gate_scale * torch.log(gate_scale + 0.00001)).sum()

        # method 1 and method 3, entropy becomes very small
        # gate_mean = self.gate.mean(0) + 1e-5
        # entropy1 = (gate_mean * torch.log(gate_mean)).sum()
        #
        # # combine method2 and method3, sum(1), sum(0)
        # entropy_list = (self.gate * torch.log(gate + 1e-5)).sum(0)
        # entropy2 = entropy_list.mean()
        #
        # sd = self.gate.std(0).sum()
        # # va = self.gate.var(0).sum()
        # entropy_norm = sd * 10 - entropy1 - entropy2

        # dis = self.gate * self.gate.T
        # gate_softmax_scale = F.softmax(self.gate, dim=0)
        # entropy_norm = (gate_softmax_scale * torch.log(gate_softmax_scale)).sum()

        #
        # # method 4, minus beta
        # gate_scale = self.gate / self.gate.sum(0)
        # gate_scale[gate_scale != gate_scale] = 0
        # expert_entropy = (gate_scale * torch.log(gate_scale)).sum(0)
        # gate_softmax_scale = F.softmax(self.gate, dim=0)
        # expert_entropy = (gate_softmax_scale * torch.log(gate_softmax_scale)).sum(0)
        # entropy_norm = (self.gate.mean(0) * expert_entropy).sum()

        # weight decay bigger or smaller

        # if self.flag == True:
        #     # print(gate)
        #     _dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'mainstream_scores')
        #     _file = os.path.join(_dir, 'gate_output_ms')
        #     if not os.path.exists(_file):
        #         os.mkdir(_file)
        #     _des = os.path.join(_file, str(self.num_experts) + '_' + self.time + '_tau_' + str(self.tau) + '_gate_output_new.npy')
        #     with open(_des, "wb") as f:
        #         np.save(f, gate.detach().cpu().numpy())

        """ Multiplying gates and experts"""
        # if early_train:
        #     self.gate = self.gate_kernels[batch_idx]
        #     res = torch.matmul(torch.reshape(self.gate, [-1, 1, self.num_experts]), expert_outputs)
        #     final_outputs = torch.reshape(res, [-1, dimension])
        #
        #     if self.training:
        #         # kl loss with gate, mean or sum
        #         weighted_kl_loss = torch.matmul(self.gate, torch.stack(kl_loss_list)).sum()
        #         return final_outputs, weighted_kl_loss
        #     else:
        #         return final_outputs
        # else:

        # loss = loss / loss.sum(1).reshape(-1, 1)
        # loss = torch.FloatTensor(loss).to(self.device)
        # self.loss_[batch_idx] = loss
        loss = F.softmax(loss, dim=1)
        self.gate[batch_idx] = loss
        res = torch.matmul(torch.reshape(loss, [-1, 1, self.num_experts]), expert_outputs)
        final_outputs = torch.reshape(res, [-1, dimension])

        # # gate = gate_outputs[0]
        # # batch * expert
        # # 512 * 1 * expert, expert: 512 * expert * dimension
        # res = torch.matmul(torch.reshape(self.gate, [-1, 1, self.num_experts]), expert_outputs)
        # final_outputs = torch.reshape(res, [-1, dimension])

        # final_outputs = expert_outputs.mean(1) # average experts outputs

        # vali_acc = torch.FloatTensor(vali_acc).to(self.device)
        # # vali_acc = vali_acc / vali_acc.sum(1).reshape(-1, 1)
        # vali_acc = F.softmax(vali_acc, dim=1)
        # res = torch.matmul(torch.reshape(vali_acc, [-1, 1, self.num_experts]), expert_outputs)
        # final_outputs = torch.reshape(res, [-1, dimension])

        # # expert dropout
        # dis_ = F.softmax(dis_pre / 0.8, dim=1) # > 1, < 1
        # indices = dis_.multinomial(num_samples=int(self.num_experts * 0.2))
        # drop_out = torch.ones(dis_pre.shape).to(self.device)
        # for i, idx in enumerate(indices):
        #     drop_out[i][idx] = 0
        # dis_pre = 1 / dis_pre
        # dis = (torch.exp(dis_pre) / torch.exp(dis_pre * drop_out).sum(1).reshape(-1, 1)) * drop_out
        # # dis = dis_pre / (dis_pre * drop_out).sum(1).reshape(-1, 1) * drop_out
        # res = torch.matmul(torch.reshape(dis, [-1, 1, self.num_experts]), expert_outputs)
        # final_outputs = torch.reshape(res, [-1, dimension])

        # # drop 10 or 20 worst, try softmax
        # # vals, indices = torch.topk(dis_pre, self.num_experts - 10)
        # vals, indices = torch.topk(dis_pre, int(self.num_experts * 0.9))
        # drop_out = torch.zeros(dis_pre.shape).to(self.device)
        # for i, idx in enumerate(indices):
        #     drop_out[i][idx] = 1
        # dis_pre = 1 / dis_pre
        # dis = (torch.exp(dis_pre) / torch.exp(dis_pre * drop_out).sum(1).reshape(-1, 1)) * drop_out
        # # dis = dis_pre / (dis_pre * drop_out).sum(1).reshape(-1, 1) * drop_out
        # res = torch.matmul(torch.reshape(dis, [-1, 1, self.num_experts]), expert_outputs)
        # final_outputs = torch.reshape(res, [-1, dimension])

        if self.training:
            # kl loss with gate, mean or sum
            # weighted_kl_loss = torch.matmul(self.gate, torch.stack(kl_loss_list)).sum()
            # weighted_kl_loss = torch.stack(kl_loss_list).mean()
            weighted_kl_loss = torch.matmul(loss, torch.stack(kl_loss_list)).sum()
            # weighted_kl_loss = torch.matmul(dis, torch.stack(kl_loss_list)).sum()
            # weighted_kl_loss = torch.matmul(vali_acc, torch.stack(kl_loss_list)).sum()
            # return final_outputs, weighted_kl_loss, entropy_norm
            return final_outputs, weighted_kl_loss
            # return final_outputs, weighted_kl_loss, gate_loss
        else:
            return final_outputs

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        # here should train each expert
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        # self.evaluator_vali = copy.deepcopy(evaluator)

        # prepare dataset
        # dataset.set_eval_data('valid')
        users = np.arange(self.num_users)
        similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'mainstream_scores')
        if not os.path.exists(similarity_dir):
            os.mkdir(similarity_dir)

        s_file = os.path.join(similarity_dir, 'avg_weights')
        if not os.path.exists(s_file):
            os.mkdir(s_file)
        s_path = os.path.join(s_file, 'weights_' + strftime('%Y%m%d-%H%M'))
        if not os.path.exists(s_path):
            os.mkdir(s_path)

        # t_file = os.path.join(similarity_dir, 'std_weights')
        # if not os.path.exists(t_file):
        #     os.mkdir(t_file)
        # t_path = os.path.join(t_file, 'std_' + strftime('%Y%m%d-%H%M'))
        # if not os.path.exists(t_path):
        #     os.mkdir(t_path)

        # gate or vali loss here
        u1_file = os.path.join(similarity_dir, 'gate_vals')
        if not os.path.exists(u1_file):
            os.mkdir(u1_file)
        u1_path = os.path.join(u1_file, 'gate_dis_' + strftime('%Y%m%d-%H%M'))
        if not os.path.exists(u1_path):
            os.mkdir(u1_path)

        # u2_file = os.path.join(similarity_dir, 'loss_vals')
        # if not os.path.exists(u2_file):
        #     os.mkdir(u2_file)
        # u2_path = os.path.join(u2_file, 'loss_dis_' + strftime('%Y%m%d-%H%M'))
        # if not os.path.exists(u2_path):
        #     os.mkdir(u2_path)

        similarity_file = os.path.join(similarity_dir, 'moe_folder_f')
        if not os.path.exists(similarity_file):
            os.mkdir(similarity_file)
        # path_ = os.path.join(similarity_file, 'MOE_scores_' + strftime('%Y%m%d-%H%M') + '_tau_' + str(self.tau))
        path_ = os.path.join(similarity_file, 'MOE_scores_' + strftime('%Y%m%d-%H%M'))
        if not os.path.exists(path_):
            os.mkdir(path_)

        # _dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'mainstream_scores')
        # _file = os.path.join(similarity_dir, 'gate_output_ms')
        # _file = os.path.join(_dir, 'avg_experts_ms')
        # if not os.path.exists(_file):
        #     os.mkdir(_file)
        # _des = os.path.join(_file, str(self.num_experts) + '_' + strftime('%Y%m%d-%H%M') + '_tau_' + str(self.tau))

        # _des = os.path.join(_file, str(self.num_experts) + '_' + strftime('%Y%m%d-%H%M'))
        # if not os.path.exists(_des):
        #     os.mkdir(_des)

        train_matrix = dataset.train_matrix.toarray()
        train_matrix = torch.FloatTensor(train_matrix).to(self.device) ###
        # train_matrix = torch.FloatTensor(train_matrix).cuda()
        best_result = None

        # for epoch
        start = time()
        ###
        # epoch_weight = None

        # # global epoch weight
        self.e_loss = torch.ones((self.num_users, 1))
        # self.e_loss /= self.num_users
        #
        bw = None

        # self.ndcg_vali = torch.ones((self.num_users, 1))
        # self.ndcg_vali /= self.num_users # for softmax

        prev_loss = None
        loss_change = None
        # loss_change_rate = None
        self.gate = torch.zeros((train_matrix.shape[0], self.num_experts)).to(self.device)
        # self.loss_ = torch.zeros((train_matrix.shape[0], self.num_experts)).to(self.device)

        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_loss = 0.0
            ce_loss_sum = 0.0
            kl_loss_sum = 0.0
            batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            # et = False


            # # global aw
            # if epoch != 1:
            #     loss_last_epoch = copy.deepcopy(self.e_loss)
            #     # implement global weight
            #     # loss_last_epoch = loss_last_epoch * self.m_dis.reshape(-1, 1)
            #     # loss_last_epoch = loss_last_epoch * self.weight.reshape(-1, 1)
            #     # alpha = 2 # 80 - 1000, 0.1->37
            #     # beta = torch.zeros(loss_last_epoch.reshape(-1, 1).shape)
            #     # bw = torch.max(((loss_last_epoch - loss_last_epoch.mean() + 2 * alpha) / (2 * alpha)).reshape(-1, 1), beta)
            #     bw = F.softmax(loss_last_epoch / 1300, dim=0) # 100 - 1500, 1 - 10
            #     # bw = F.tanh(loss_last_epoch / loss_last_epoch.mean())
            #
            #     # ndcg_last_epoch = copy.deepcopy(epoch_weight).reshape(-1, 1)
            #     # alpha = 0.4 # 0.08 - 1
            #     # beta = torch.zeros(ndcg_last_epoch.reshape(-1, 1).shape)
            #     # ndcg_last_epoch = ndcg_last_epoch * self.m_dis.reshape(-1, 1)
            #     # ndcg_last_epoch = ndcg_last_epoch * self.weight.reshape(-1, 1)
            #     # bw = torch.max(((ndcg_last_epoch - ndcg_last_epoch.mean() + 2 * alpha) / (2 * alpha)).reshape(-1, 1), beta)
            #     # bw = F.softmax(ndcg_last_epoch / 0.05, dim=0) # 0.001
            #
            #     # # shift by max value
            #     # beta = 0.007
            #     # exp_out = torch.exp(beta * (loss_last_epoch - loss_last_epoch.max()))
            #     # bw = exp_out / (torch.sum(exp_out) + 1e-8)
            #
            # else:
            #     bw = copy.deepcopy(self.e_loss)
            #     # bw = copy.deepcopy(self.e_loss) * self.m_dis.reshape(-1, 1)
            #     # bw = copy.deepcopy(self.e_loss) * self.weight.reshape(-1, 1)
            #     # bw = copy.deepcopy(self.ndcg_vali)
            # bw = self.weight.reshape(-1, 1) * bw

            # run 10, 20, 30, 40 epochs first
            if epoch <= 15:
                bw = torch.ones((self.num_users, 1))
                # bw = torch.ones((self.num_users, 1)) / self.num_users
                # bw = self.weight.reshape(-1, 1)
            else:
                # loss_last_epoch = copy.deepcopy(self.e_loss)
                # alpha = 600 # 80 - 1000, 0.1->37
                # beta = torch.zeros(loss_last_epoch.reshape(-1, 1).shape)
                # bw = torch.max(((loss_last_epoch - loss_last_epoch.mean() + 2 * alpha) / (2 * alpha)).reshape(-1, 1), beta)

                loss_change_avg = loss_change.mean(1).reshape(-1, 1)
                # loss_change_avg = loss_change_rate.mean(1).reshape(-1, 1) ##
                # loss_change_avg = loss_change_avg.where(loss_change_avg > 0, torch.tensor(0.0001))
                alpha = 0.6
                # beta = torch.zeros(loss_change_avg.shape)
                beta = torch.ones(loss_change_avg.shape) * (1e-8)
                # loss_change_avg = loss_change_avg * self.m_dis.reshape(-1, 1)
                # loss_change_avg = loss_change_avg * self.weight.reshape(-1, 1)
                bw = torch.max(((loss_change_avg - loss_change_avg.mean() + 2 * alpha) / (2 * alpha)).reshape(-1, 1), beta)

                # # bw = F.softmax(loss_change_avg / 0.01, dim=0)

                # # shift by max value
                # beta = 0.5
                # exp_out = torch.exp(beta * (loss_change_avg - loss_change_avg.max()))
                # bw = exp_out / (torch.sum(exp_out) + 1e-8)
            # bw = self.weight.reshape(-1, 1) * bw

            # if epoch <= 4:
            #     bw = torch.ones((self.num_users, 1))
            #     # bw = torch.ones((self.num_users, 1)) / self.num_users
            #     # bw = self.weight.reshape(-1, 1)
            # else:
            #     loss_change_avg = loss_change.mean(1).reshape(-1, 1) ##
            #     # loss_change_avg = loss_change_avg.where(loss_change_avg > 0, torch.tensor(0.0001))
            #     alpha = 0.3
            #     beta = torch.zeros(loss_change_avg.shape)
            #     # loss_change_avg = loss_change_avg * self.m_dis.reshape(-1, 1)
            #     # loss_change_avg = loss_change_avg * self.weight.reshape(-1, 1)
            #     bw = torch.max(((loss_change_avg - loss_change_avg.mean() + 2 * alpha) / (2 * alpha)).reshape(-1, 1), beta)
            #
            #     # # bw = F.softmax(loss_change_avg / 0.01, dim=0)
            #
            #     # # shift by max value
            #     # beta = 0.5
            #     # exp_out = torch.exp(beta * (loss_change_avg - loss_change_avg.max()))
            #     # bw = exp_out / (torch.sum(exp_out) + 1e-8)
            # bw = self.weight.reshape(-1, 1) * bw

            # # 70 for ml-1m, see the distribution
            # if epoch == 35 or epoch == 40:
            #     print(bw.min(), bw.max(), bw.mean(), bw.std())

            for b, batch_idx in enumerate(batch_loader):
                batch_matrix = train_matrix[batch_idx]

                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap

                # if epoch <= 5:
                #     et = True
                #     batch_loss, ce, kl = self.train_model_per_batch(batch_matrix, batch_idx, batch_weight=None, early_train=True)
                #     # batch_weight = torch.FloatTensor(self.weight[batch_idx]).to(self.device)
                #     # batch_loss, ce, kl = self.train_model_per_batch(batch_matrix, batch_idx, batch_weight, early_train=True)
                # else:
                #     et = False
                #     # batch_loss, ce, kl = self.train_model_per_batch(batch_matrix, batch_idx, batch_weight=None, early_train=et)

                # # normal
                # # weighted loss
                # batch_weight = torch.FloatTensor(self.weight[batch_idx]).to(self.device)
                # batch_loss, ce, kl = self.train_model_per_batch(batch_matrix, batch_idx, batch_weight)

                # batch_loss, ce, kl = self.train_model_per_batch(batch_matrix, batch_idx)

                # epoch weight
                batch_weight = bw[batch_idx].to(self.device) ###
                # batch_weight = bw[batch_idx].cuda()
                batch_loss, ce, kl = self.train_model_per_batch(batch_matrix, batch_idx, batch_weight)

                # if epoch == 1:
                #     batch_loss, ce, kl = self.train_model_per_batch(batch_matrix, batch_idx, batch_weight)
                # else:
                #     batch_loss, ce, kl = self.train_model_per_batch(batch_matrix, batch_idx, batch_weight[batch_idx].to(self.device))

                # # vali_metric
                # if b == 0 and epoch == 1:
                #     batch_loss, ce, kl = self.train_model_per_batch(batch_matrix, batch_idx, batch_weight)
                # else:
                #     batch_loss, ce, kl = self.train_model_per_batch(batch_matrix, batch_idx, torch.FloatTensor(np.array(batch_weight['NDCG@50'])[batch_idx]).to(self.device))

                epoch_loss += batch_loss
                ce_loss_sum += ce
                kl_loss_sum += kl

                # self.eval()
                # test_score, batch_weight = evaluator.evaluate_vali_MOE(self)

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            # self.alpha *= 0.92

            # calculate loss change, or 4 epoch (3 changes), 3 epoch (2 changes), less or equal
            if epoch <= 4:
                if epoch == 1:
                    pass
                elif epoch == 2:
                    # loss_change = self.e_loss - prev_loss
                    loss_change = prev_loss - self.e_loss
                else:
                    # loss_change = torch.cat((loss_change, self.e_loss - prev_loss), 1)
                    loss_change = torch.cat((loss_change, prev_loss - self.e_loss), 1)
                    # loss_change_rate = loss_change / loss_change.mean() ##
            else:
                # loss_change = torch.cat((loss_change[1:], self.e_loss - prev_loss), 1)
                # cur_window_avg = loss_change.mean() ##
                loss_change = torch.cat((loss_change[:, 1:], prev_loss - self.e_loss), 1)
                # loss_change_rate = loss_change / cur_window_avg ##

            prev_loss = copy.deepcopy(self.e_loss)

            # epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]
            epoch_info = ['epoch=%3d' % epoch, 'epoch loss=%.3f' % epoch_loss, 'ce loss=%.3f' % ce_loss_sum, 'kl loss=%.3f' % kl_loss_sum, 'train time=%.2f' % epoch_train_time]

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                # # normal
                test_score = evaluator.evaluate_vali(self)

                # # epoch vali ndcg
                # test_score, test_score_all = evaluator.evaluate_vali_MOE(self)
                # epoch_weight = torch.FloatTensor(np.array(test_score_all['NDCG@100']))

                # test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]

                updated, should_stop = early_stop.step(test_score, epoch)
                test_score_output = evaluator.evaluate(self)
                test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output]
                # output = ['%s=%.4f' % (k, test_score[k]) for k in test_score]
                # print("vali set:", output)
                # _, _ = self.es.step(test_score_output, epoch)


                # used to draw graph for 5 groups of user changes
                with open(os.path.join(s_path, str(epoch) + '_epoch.npy'), 'wb') as f:
                    np.save(f, bw.detach().cpu().numpy())

                # with open(os.path.join(t_path, str(epoch) + '_epoch.npy'), 'wb') as f:
                #     np.save(f, bw.std())


                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        # # save memory, do not save the best parameters
                        # torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))

                        # rec = self.predict_all()
                        # self.flag = True
                        best_result = test_score_output
                        ndcg_test_all = evaluator.evaluate(self, mean=False) ##
                        # self.flag = False

                        # with open(os.path.join(_des, 'gate_output_dis.npy'), "wb") as f:
                        #     np.save(f, self.gate_kernels[self.test_gate_ids].detach().cpu().numpy())

                        ## train the gate
                        # with open(os.path.join(_des, 'gate_output_dis.npy'), "wb") as f:
                        #     np.save(f, self.gate.detach().cpu().numpy())


                        ## get gates vals
                        with open(os.path.join(u1_path, 'gate_output_dis.npy'), "wb") as f:
                            np.save(f, self.gate.detach().cpu().numpy())

                        # with open(os.path.join(u2_path, 'loss_output_dis.npy'), "wb") as f:
                        #     np.save(f, self.loss_.detach().cpu().numpy())

                        with open(os.path.join(path_, str(self.num_experts) + '_moe_test_scores.npy'), 'wb') as f:
                            np.save(f, ndcg_test_all)

                        if self.anneal_cap == 1: print(self.anneal)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                # epoch_info = ['epoch=%3d' % epoch, 'epoch loss=%.3f' % epoch_loss, 'ce loss=%.3f' % ce_loss_sum, 'kl loss=%.3f' % kl_loss_sum, 'train time=%.2f' % epoch_train_time]
                epoch_info += test_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        # save experts distribution
        # for i in range(self.num_experts):
        #     with open(os.path.join(_des, str(i) + '_avg_experts_dis.npy'), "wb") as f:
        #         ndcg_temp = evaluator.evaluate_all(self.expert_kernels[i](train_matrix)[0].detach().cpu().numpy(), mean=False)['NDCG@20']
        #         np.save(f, ndcg_temp)

        total_train_time = time() - start

        # self.flag = True
        # gate_test = self.predict(self.test_gate_ids, dataset.train_matrix)
        # self.flag = False
        # return early_stop.best_score, total_train_time
        # return self.es.best_score, total_train_time
        return best_result, total_train_time

    def train_model_per_batch(self, batch_matrix, batch_idx, batch_weight=None):
        # zero grad
        self.optimizer.zero_grad()
        # alpha = 0.02
        mask = 1 - batch_matrix

        # model forwrad
        # output, kl_loss, entropy_norm = self.forward(batch_matrix)
        # output, kl_loss, gate_loss = self.forward(batch_matrix, batch_idx)
        output, kl_loss = self.forward(batch_matrix, batch_idx)

        vali_loss = -torch.log(torch.exp(output) / torch.exp(output * mask).sum(1).reshape(-1, 1))
        expert_vali_loss = (vali_loss * self.vali_batch).sum(1)
        # expert_vali_loss = (vali_loss * self.vali_batch_cuda).sum(1)

        # # training loss
        # expert_training_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1)

        # beta = torch.ones(expert_training_loss.reshape(-1, 1).shape).to(self.device)  ###
        # # 0.1 to 1
        # beta *= 0.01

        # beta = torch.zeros(expert_training_loss.reshape(-1, 1).shape).to(self.device)
        # batch_weight = torch.max(((expert_training_loss - expert_training_loss.mean() + 2 * alpha) / (2 * alpha)).reshape(-1, 1), beta)
        # batch_weight = F.softmax(expert_training_loss / 5000) # 1000 - 5000

        # batch_weight = torch.max(((expert_vali_loss - expert_vali_loss.mean() + 2 * alpha) / (2 * alpha)).reshape(-1, 1), beta) # 0.01
        # batch_weight = batch_weight / batch_weight.max()
        # batch_weight = batch_weight / batch_weight.mean()
        # batch_weight = torch.max(((expert_vali_loss - expert_vali_loss.mean() + 2 * self.alpha) / (2 * self.alpha)).reshape(-1, 1), beta)

        # batch_weight = F.softmax(expert_vali_loss / 1000) ## 50 - 1000, 10000

        self.e_loss[batch_idx] = expert_vali_loss.reshape(-1, 1).detach().cpu() ### update batch weights

        # shift by max value
        # beta = 0.1
        # exp_out = torch.exp(beta * (expert_vali_loss - expert_vali_loss.max()))
        # batch_weight = exp_out / (torch.sum(exp_out) + 1e-8)

        # # tanh, mean / max
        # batch_weight = F.tanh(expert_vali_loss / expert_vali_loss.max()) ###
        # # batch_weight = torch.atan(expert_vali_loss / expert_vali_loss.mean())

        # tanh + normalization
        # gamma = 1.5
        # weight = (F.tanh(expert_vali_loss / expert_vali_loss.max())) ** gamma
        # batch_weight = weight / weight.sum()
        # weight = (torch.atan(expert_vali_loss / expert_vali_loss.max())) ** gamma
        # batch_weight = weight / weight.sum()

        # # tahn + softmax tau
        # weight = (F.tanh(expert_vali_loss / expert_vali_loss.max()))
        # batch_weight = F.softmax(weight / 0.5)

        # x = expert_vali_loss.detach().cpu().numpy()
        # y = batch_weight.detach().cpu().numpy()
        #
        # plt.bar(x, y)
        # plt.show()

        if batch_weight is None:
            ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()
            # ce_loss = -((F.log_softmax(output, 1) * batch_matrix).sum(1) * entropy_norm).mean()
            # experts_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).reshape(-1, 1)
            # ce_loss = ((experts_loss * gate_loss).sum(0) / gate_loss.sum(0)).sum()
        else:
            # print("batch_weight")
            # normalization
            # batch_weight = 1 - batch_weight
            # batch_weight = 1 / batch_weight.where(batch_weight != 0, min(filter(lambda i: i > 0, batch_weight)))
            # alpha = 80 # 0.1 - 1 (50), 0.22 - 2 (100); epoch 0.25 - 1;;; 0.06 - 1
            # beta = torch.zeros(batch_weight.reshape(-1, 1).shape).to(self.device)
            # batch_weight = torch.max(((batch_weight - batch_weight.mean() + 2 * alpha) / (2 * alpha)).reshape(-1, 1), beta)

            # batch_weight = F.softmax(batch_weight / 30) # 0.03 - 1 (batch); 0.1 - 1, 2;; 1 - 30
            # batch_weight = F.tanh(batch_weight / batch_weight.max())
            # batch_weight = torch.atan(batch_weight / batch_weight.max())
            ce_loss = -((F.log_softmax(output, 1) * batch_matrix) * batch_weight.view(output.shape[0], -1)).sum(1).sum() / batch_weight.sum()
            # ce_loss = -((F.log_softmax(output, 1) * batch_matrix).sum(1) * batch_weight).sum() / batch_weight.sum()
            # ce_loss = -((F.log_softmax(output, 1) * batch_matrix).sum(1) * batch_weight).sum()
        # loss = ce_loss + kl_loss * self.anneal + entropy_norm * self.beta # for method 1
        # loss = ce_loss + kl_loss * self.anneal - entropy_norm * self.beta  # for method 3 and 4
        loss = ce_loss + kl_loss * self.anneal

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        self.update_count += 1

        # return loss, ce_loss, kl_loss, entropy_norm
        return loss, ce_loss, kl_loss

    # def predict_all(self):
    #     R = self.predict_for_eval(np.arange(self.num_users))
    #     return R
    #
    # def predict_for_eval(self, user_ids):
    #     self.eval()
    #     batch_eval_pos = self.dataset.train_matrix[user_ids]
    #     with torch.no_grad():
    #         eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
    #         eval_output = self.forward(eval_input).detach().cpu().numpy()
    #     self.train()
    #     return eval_output

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        # simply multiple here
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device) ###
            # eval_input = torch.Tensor(batch_eval_pos.toarray()).cuda()
            eval_output = self.forward(eval_input, user_ids).detach().cpu().numpy()

            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output

class gate_network(nn.Module):
    def __init__(self, num_items, num_experts):
        super(gate_network, self).__init__()
        self.num_items = num_items
        self.num_experts = num_experts
        # one layer
        # proj_dim = 1024
        # # self.fc = nn.Linear(self.num_items, self.num_experts)
        # init_t = 0.5
        # self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        # self.cosine_projector = torch.nn.Linear(self.num_items, proj_dim)
        # self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, self.num_experts)), requires_grad=True)
        # self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item() #try 0.001
        # torch.nn.init.normal_(self.sim_matrix, 0, 0.01)

        ## two layers
        # # hidden_size = int((2/3) * (num_items + num_experts))
        hidden_size = 2000
        self.fc1 = nn.Linear(self.num_items, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.num_experts)

        # ## three layers
        # # hidden_size_2 = int((2 / 3) * (hidden_size + num_experts))
        # hidden_size_2 = 1000
        # self.fc2 = nn.Linear(hidden_size, hidden_size_2)
        # self.fc3 = nn.Linear(hidden_size_2, self.num_experts)

        # self.fc4 = nn.Linear(500, 250)
        #
        # self.fc5 = nn.Linear(250, self.num_experts)

        # self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## one layer
        # output = self.fc(x)

        # two layers
        hidden = self.fc1(x)
        # hidden = self.dropout(hidden)
        relu = self.relu(hidden)
        output = self.fc2(relu)

        # ## three layers
        # hidden1 = self.fc1(x)
        # # hidden = self.dropout(hidden)
        # relu1 = self.sigmoid(hidden1)
        # hidden2 = self.fc2(relu1)
        # relu2 = self.relu(hidden2)
        # output = self.fc3(relu2)

        # output = self.relu(output)
        # output = self.fc4(output)
        # output = self.relu(output)
        # output = self.fc5(output)

        # cosine_p = self.cosine_projector
        # sim_mat = self.sim_matrix
        #
        # logits = torch.matmul(F.normalize(cosine_p(x), dim=1), F.normalize(sim_mat, dim=0))
        # logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        # output = logits * logit_scale

        # return
        output = self.softmax(output)
        return output


class MultVAE_Expert(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MultVAE_Expert, self).__init__(dataset, model_conf)
        self.num_items = dataset.num_items

        self.enc_dims = [self.num_items] + model_conf['enc_dims']
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.total_anneal_steps = model_conf['total_anneal_steps']
        self.anneal_cap = model_conf['anneal_cap']

        self.dropout = model_conf['dropout']
        self.reg = model_conf['reg']

        self.lr = model_conf['lr']

        self.device = device

        self.build_graph()

    def build_graph(self):
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            # linear_layer = nn.Linear(d_in, d_out)
            # nn.init.uniform_(linear_layer.weight, -0.1, 0.1)
            # self.encoder.append(linear_layer)
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            # linear_layer = nn.Linear(d_in, d_out)
            # nn.init.uniform_(linear_layer.weight, -0.1, 0.1)
            # self.decoder.append(linear_layer)
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.Tanh())

        # optimizer
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device) ###
        # self.cuda()

    def forward(self, x):
        # encoder
        h = F.dropout(F.normalize(x), p=self.dropout, training=self.training)
        for layer in self.encoder:
            h = layer(h)

        # sample
        mu_q = h[:, :self.enc_dims[-1]]
        logvar_q = h[:, self.enc_dims[-1]:]  # log sigmod^2  batch x 200
        std_q = torch.exp(0.5 * logvar_q)  # sigmod batch x 200

        # F.kl_div()

        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=0.01)
        sampled_z = mu_q + self.training * epsilon * std_q

        output = sampled_z
        for layer in self.decoder:
            output = layer(output)

        if self.training:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return output, kl_loss
        else:
            return output

    # def predict(self, user_ids, eval_pos_matrix, eval_items=None):
    #     self.eval()
    #     batch_eval_pos = eval_pos_matrix[user_ids]
    #     with torch.no_grad():
    #         eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
    #         eval_output = self.forward(eval_input).detach().cpu().numpy()
    #
    #         if eval_items is not None:
    #             eval_output[np.logical_not(eval_items)] = float('-inf')
    #         else:
    #             eval_output[batch_eval_pos.nonzero()] = float('-inf')
    #     self.train()
    #     return eval_output