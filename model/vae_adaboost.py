import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from utils import Tool
from utils import MP_Utility

import copy
from past.builtins import range

import pickle
import argparse
import pandas as pd
from scipy.sparse import csr_matrix, rand as sprand
from tqdm import tqdm
from time import strftime

# from statistics import mean
from scipy.special import softmax
from scipy.special import log_softmax
from concurrent.futures import ThreadPoolExecutor

# np.random.seed(0)
# torch.manual_seed(0)

class vae_adaboost(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(vae_adaboost, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_df = dataset.train_df
        # self.stumps = None
        # self.stump_weights = None
        # self.errors = None
        # self.sample_weights = None
        # self.ada_errors = None

        self.iters = model_conf['iters']
        self.neg_sample_num = model_conf['neg_sample_num']
        self.neg_sample_rate_eval = model_conf['neg_sample_rate_eval']
        self.beta1 = model_conf['beta1']
        self.beta2 = model_conf['beta2']
        # self.l = model_conf['lambda']
        self.tau = model_conf['tau']
        self.model_conf = model_conf
        self.device = device
        help_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name)
        help_dir = os.path.join(help_dir, 'bias_scores')
        self.test_like_item = np.load(help_dir + '/test_like_item.npy', allow_pickle=True)

        # self.test_data = dataset.test_dict
        # self.test_ = np.zeros((self.num_users, self.num_items))
        # for u in self.test_data:
        #     self.test_[u][self.test_data[u]] = 1
        #
        # self.vali_data = dataset.vali_dict
        # self.vali_ = np.zeros((self.num_users, self.num_items))
        # for u in self.vali_data:
        #     self.vali_[u][self.vali_data[u]] = 1

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']
        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir
        # users = np.arange(self.num_users)

        self.time = strftime('%Y%m%d-%H%M')
        similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
        if not os.path.exists(similarity_dir):
            os.mkdir(similarity_dir)

        s_file = os.path.join(similarity_dir, 'vae_adaboost_records')
        if not os.path.exists(s_file):
            os.mkdir(s_file)
        similarity_file = os.path.join(s_file, self.time + '_vae_record_scores')
        if not os.path.exists(similarity_file):
            os.mkdir(similarity_file)
        best_result = None

        start = time()
        m = self.num_users
        n = self.num_items
        # initlizing numpy arrays
        # self.sample_weights = np.zeros(shape=(self.iters, m))
        # self.stumps = np.zeros(shape=self.iters, dtype=object)
        # self.stump_weights = np.zeros(shape=self.iters)
        # self.errors = np.zeros(shape=self.iters)
        # self.ada_errors = np.zeros(shape=self.iters)
        # self.sample_weights = []
        self.stumps = []
        # self.stump_weights = []
        # self.errors = []
        # self.ada_errors = []
        self.user_err_means_list = []
        self.item_err_means_list = []

        # initializing weights uniformly
        # how to initialize sample weights and put it into
        # self.sample_weights[0] = np.ones(shape=m) / m
        # # self.sample_weights[0] = np.ones(shape=(m, n)) / n
        # self.sample_weights_users = np.ones(shape=(m, 1)) / m
        # self.sample_weights_items = np.ones(shape=(n, 1)) / n

        # self.sample_weights.append(np.ones(shape=(m, n)))
        # self.sample_weights.append(np.ones(shape=(m, n)) / (m * n)) # desgin one

        self.sample_weights = np.ones(shape=(m, n)) / (m * n)
        # local_stump = MF(dataset, self.model_conf, self.device)
        train_matrix = dataset.train_matrix.toarray()
        self.train_matrix = torch.FloatTensor(train_matrix).to(self.device)

        for t in tqdm(range(self.iters)):
            # fitting weak learner
            es = copy.deepcopy(early_stop)
            # curr_sample_weights = self.sample_weights[t]

            curr_sample_weights = self.sample_weights
            # cur_sample_users = self.sample_weights_users
            # cur_sample_items = self.sample_weights_items
            stump = MultVAE(dataset, self.model_conf, self.device)
            stump.train_model(dataset, evaluator, es, logger, config, similarity_file,
                              curr_sample_weights * (self.num_users * self.num_items))
            # stump.train_model(dataset, evaluator, es, logger, config, similarity_file, curr_sample_weights)

            # # ndcg@20 for weight initialization
            # # calculating error and stump weight from weak learner prediction
            # # calculate errors via mse
            # mask = 1 - self.test_ - self.vali_
            # stump_pred = stump.get_rec() * mask # for testing data mse
            # y = dataset.train_matrix.toarray()
            # stump_pred = stump.get_rec()
            # err = np.square(stump_pred - y) # square error, zeros will become nan in the later operations

            # y = dataset.train_matrix.toarray()
            # stump_pred = stump.get_rec()
            # stemp_sig = 1 / (1 + np.exp(-stump_pred))
            # err = -(y * np.log(stemp_sig) + (1 - y) * np.log(1 - stemp_sig)) # binary cross entropy


            ### output, kl_loss
            rec = stump.forward(self.train_matrix)[0].detach().cpu().numpy()
            user_err_list, item_err_list = [], []
            for i in range(self.neg_sample_num):
                # temp_users = [[] for _ in range(self.num_users)]
                # temp_items = [[] for _ in range(self.num_items)]
                # user_list, item_list, label_list = MP_Utility.negative_sampling(self.num_users,
                #                                                                 self.num_items,
                #                                                                 self.train_df[0],
                #                                                                 self.train_df[1],
                #                                                                 self.neg_sample_rate_eval)

                user_list, item_list, label_list, rec_list = MP_Utility.negative_sampling_vae(self.num_users,
                                                                                              self.num_items,
                                                                                              self.train_df[0],
                                                                                              self.train_df[1],
                                                                                              self.neg_sample_rate_eval,
                                                                                              rec)


                sig_rec = 1 / (1 + np.exp(-rec_list)) # sigmoid
                total_err = -(label_list * np.log(sig_rec) + (1 - label_list) * np.log(1 - sig_rec))

                # # squared error
                # total_err = np.square(sig_rec - label_list)
                # # total_err = np.square(rec_list - label_list)

                # rec_pos = (u_pos * v_pos).sum(axis=1)
                # rec_neg = (u_neg * v_neg).sum(axis=1)
                #
                # sig_rec_pos = 1 / (1 + np.exp(-rec_pos))
                # sig_rec_neg = 1 / (1 + np.exp(-rec_neg))
                # pos_err = -(pos_label * np.log(sig_rec_pos) + (1 - pos_label) * np.log(1 - sig_rec_pos))
                # neg_err = -(neg_label * np.log(sig_rec_neg) + (1 - neg_label) * np.log(1 - sig_rec_neg))
                #
                # user_list = np.concatenate([pos_user, neg_user], axis=0)
                # item_list = np.concatenate([pos_item, neg_item], axis=0)
                # pos_err = self.l * pos_err
                # neg_err = (1 - self.l) * neg_err
                # total_err = np.concatenate([pos_err, neg_err], axis=0)

                # for idx, val in enumerate(total_err):
                #     temp_users[idx].append(val)
                #     temp_items[idx].append(val)
                # user_err.append(temp_users.mean(axis=1))
                # item_err.append(temp_items.mean(axis=1))
                user_err = np.zeros(self.num_users)
                item_err = np.zeros(self.num_items)
                np.add.at(user_err, user_list, total_err)
                np.add.at(item_err, item_list, total_err)

                user_err /= np.bincount(user_list)
                item_err /= np.bincount(item_list)
                user_err_list.append(user_err)
                item_err_list.append(item_err)

            user_err_mean = np.mean(user_err_list, axis=0).reshape(-1, 1)
            item_err_mean = np.mean(item_err_list, axis=0).reshape(-1, 1)

            # item_err_mean[np.isnan(item_err_mean)] = 0  ## for positive sample
            self.user_err_means_list.append(user_err_mean)
            self.item_err_means_list.append(item_err_mean)

            # # save
            # self.u_file = os.path.join(similarity_file, 'user_vectors')
            # if not os.path.exists(self.u_file):
            #     os.mkdir(self.u_file)
            #
            # self.i_file = os.path.join(similarity_file, 'item_vectors')
            # if not os.path.exists(self.i_file):
            #     os.mkdir(self.i_file)
            #
            # with open(os.path.join(similarity_file, t+'_u.npy'), 'wb') as f:
            #     np.save(f, user_err_mean)
            #
            # with open(os.path.join(similarity_file, t+'_i.npy'), 'wb') as f:
            #     np.save(f, item_err_mean)


            err = np.matmul(user_err_mean, item_err_mean.T)
            # clip
            np.clip(err, 1e-15, 1 - 1e-15)


            # err = np.matmul(np.array(user_err_list).T, np.array(item_err_list)) ## check this
            # np.clip(err, 1e-15, 1 - 1e-15)

            # err[err == 0] = 1e-10 # replacing all zeros into a small value, squared error
            ###

            # # stump_pred = stump.predict(np.arange(self.num_users), dataset.train_matrix)
            # stump_pred = stump.get_rec()
            # y = self.test_
            # err = curr_sample_weights[(stump_pred != y)].sum() / n

            # stump_weight = np.log(1 - (curr_sample_weights * err).sum()) # design 1
            # print(stump_weight + 1)
            stump_weight = np.log((self.beta1) / (self.beta2 + err)) # design 2
            # stump_weight = (stump_weight - np.min(stump_weight)) / (np.max(stump_weight) - np.min(stump_weight))

            # stump_weight = np.exp(-self.beta1 * err) # beta here bigger, the difference is higher.   if err is high, w should be low
            # stump_weight = (stump_weight / stump_weight.sum()) * (self.num_users * self.num_items)
            # stump_weight = np.ones(shape=(m, n))

            # updating sample weights
            # correct performance higher, next_cur_weight low
            # new_sample_weights = (
            #         curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
            # )
            # new_sample_weights = curr_sample_weights * np.exp(self.beta2 * err)
            new_sample_weights = curr_sample_weights * np.exp(stump_weight * err)
            # new_sample_weights = curr_sample_weights * (1 - np.exp(-self.beta2 * err))
            # new_sample_weights = (new_sample_weights / new_sample_weights.sum()) * (self.num_users * self.num_items) # if err is high, n_w should be high
            new_sample_weights = (new_sample_weights / new_sample_weights.sum()) # sum is 1
            # new_sample_weights = np.ones(shape=(m, n))

            # how to normalization, user-item consider
            # new_sample_weights /= new_sample_weights.sum(axis=1).reshape((-1,1))
            # new_sample_weights /= new_sample_weights.sum()

            # new_sample_user_weight = cur_sample_users * ndcg20
            # new_sample_item_weight = cur_sample_items * mdg20

            # new_sample_user_weight /= new_sample_user_weight.sum(axis=1)
            # new_sample_item_weight /= new_sample_item_weight.sum(axis=1)

            # # add temperature maybe
            # new_sample_user_weight = softmax(new_sample_user_weight, axis=1)
            # new_sample_item_weight = softmax(new_sample_item_weight, axis=1)


            # # updating sample weights for t+1
            # if t + 1 < self.iters:
            #     # self.sample_weights[t + 1] = new_sample_weights
            #     self.sample_weights.append(new_sample_weights)
            #     # self.sample_weights_users = new_sample_user_weight
            #     # self.sample_weights_items = new_sample_item_weight

            self.sample_weights = new_sample_weights

            # self.stumps[t] = stump
            # self.stump_weights[t] = stump_weight
            # self.errors[t] = err
            # self.ada_errors[t] = np.prod(((self.errors[t] * (1 - self.errors[t])) ** 1 / 2))
            self.stumps.append(stump)

            # self.stump_weights.append(stump_weight) ## memory consuming ######

            # self.errors.append(err)
            # torch.cuda.empty_cache()
            # self.ada_errors.append(np.prod(((self.errors[t] * (1 - self.errors[t])) ** 1 / 2)))

        test_score_output = evaluator.evaluate(self)
        # test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output]
        # test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output if k.startswith('NDCG')]

        ndcg_test_all = evaluator.evaluate(self, mean=False)
        vae_boost_file = os.path.join(similarity_dir, 'vae_adaboost_scores')
        if not os.path.exists(vae_boost_file):
            os.mkdir(vae_boost_file)
        with open(os.path.join(vae_boost_file, self.time + '_boost_scores.npy'), 'wb') as f:
            np.save(f, ndcg_test_all)

        # # save weights and err
        # w_dir = os.path.join(similarity_dir, 'saves')
        # if not os.path.exists(w_dir):
        #     os.mkdir(w_dir)
        # w_file = os.path.join(w_dir, self.time + '_weights')
        # if not os.path.exists(w_file):
        #     os.mkdir(w_file)
        # with open(os.path.join(w_file, 'err.npy'), 'wb') as f:
        #     np.save(f, self.errors[5])
        # with open(os.path.join(w_file, 'alpha.npy'), 'wb') as f:
        #     np.save(f, self.stump_weights[5])
        # with open(os.path.join(w_file, 'sample_weights.npy'), 'wb') as f:
        #     np.save(f, self.sample_weights[5])

        total_train_time = time() - start

        return test_score_output, total_train_time

    # def parallelized_task(self, args):
    #     i, stump, stump_weight, tau = args
    #     alpha = softmax(stump_weight / tau)
    #     return alpha * stump.get_rec()

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            Rec = self.predict_helper()
            eval_output = Rec[user_ids, :]
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output

    def predict_helper(self):
        # return (self.stump_weights * np.array([stump.get_rec() for stump in self.stumps])).sum(axis=0)
        # # design 1
        # rec = None
        # self.stump_weights = np.array(self.stump_weights)
        # self.stump_weights = softmax(self.stump_weights / self.tau) # design 1 with softmax as normalization
        # for i, stump in enumerate(self.stumps):
        #     if i == 0:
        #         rec = self.stump_weights[i] * stump.get_rec()
        #     else:
        #         rec += self.stump_weights[i] * stump.get_rec()
        # return rec

        # rec = None
        # for i, stump in enumerate(self.stumps):
        #     if i == 0:
        #         rec = (self.stump_weights[i] + 1) * stump.get_rec()
        #     else:
        #         rec += (self.stump_weights[i] + 1) * stump.get_rec()
        # return rec

        # # design 2
        # rec = None
        # for i, stump in enumerate(self.stumps):
        #     # alpha = (self.stump_weights[i] - np.min(self.stump_weights[i])) / (
        #     #             np.max(self.stump_weights[i]) - np.min(self.stump_weights[i]))
        #     alpha = softmax(self.stump_weights[i] / self.tau)
        #     if i == 0:
        #         rec = alpha * stump.get_rec()
        #     else:
        #         rec += alpha * stump.get_rec()
        # return rec

        rec = None
        for i, stump in enumerate(self.stumps):
            err = np.matmul(self.user_err_means_list[i], self.item_err_means_list[i].T)

            # # load save files
            # users = np.load(self.u_file + '/' + i + '_u.npy', allow_pickle=True)
            # items = np.load(self.i_file + '/' + i + '_i.npy', allow_pickle=True)
            # err = np.matmul(users, items.T)

            stump_weight = np.log((self.beta1) / (self.beta2 + err))
            alpha = softmax(stump_weight / self.tau)
            if i == 0:
                rec = alpha * stump.get_rec(self.train_matrix)
            else:
                rec += alpha * stump.get_rec(self.train_matrix)
        return rec

        # num_threads = 5
        #
        # with ThreadPoolExecutor(max_workers=num_threads) as executor:
        #     tasks = [(i, stump, self.stump_weights[i], self.tau) for i, stump in enumerate(self.stumps)]
        #     results = list(executor.map(self.parallelized_task, tasks))
        #
        # return np.sum(results, axis=0)


class MultVAE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MultVAE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.enc_dims = [self.num_items] + model_conf['enc_dims']
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.total_anneal_steps = model_conf['total_anneal_steps']
        self.anneal_cap = model_conf['anneal_cap']

        self.dropout = model_conf['dropout']
        self.reg = model_conf['reg']

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']

        self.lr = model_conf['lr']
        self.eps = 1e-6
        self.anneal = 0.
        self.update_count = 0

        self.device = device
        self.best_params = None
        # self.es = EarlyStop(10, 'mean')

        # similarity_dir = os.path.join(dataset.data_dir, dataset.data_name, 'mainstream_scores')
        # similarity_file = os.path.join(similarity_dir, 'MS_similarity.npy')
        # self.ms = np.load(similarity_file)
        # weight_temp = self.ms / np.max(self.ms)
        # self.weight = (1 / weight_temp)
        self.time = strftime('%Y%m%d-%H%M')

        self.build_graph()

    def build_graph(self):
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.Tanh())

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device)

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

    def train_model(self, dataset, evaluator, early_stop, logger, config, similarity_file, sample_weights):
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        # prepare dataset
        # dataset.set_eval_data('valid')
        users = np.arange(self.num_users)

        train_matrix = dataset.train_matrix.toarray()
        train_matrix = torch.FloatTensor(train_matrix)
        best_result = None
        # best_epoch = -1

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            # if epoch - best_epoch > 10:
            #     break
            self.train()

            epoch_loss = 0.0

            batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                batch_matrix = train_matrix[batch_idx].to(self.device)

                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap

                # weighted loss (boosting)
                batch_weight = torch.FloatTensor(sample_weights[batch_idx, :]).to(self.device)
                batch_loss = self.train_model_per_batch(batch_matrix, batch_weight)

                # batch_loss = self.train_model_per_batch(batch_matrix)

                # weighted loss
                # batch_weight = torch.FloatTensor(self.weight[batch_idx]).to(self.device)
                # batch_loss = self.train_model_per_batch(batch_matrix, batch_weight)
                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]
            similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
            if not os.path.exists(similarity_dir):
                os.mkdir(similarity_dir)

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                test_score = evaluator.evaluate_vali(self)
                updated, should_stop = early_stop.step(test_score, epoch)

                test_score_output = evaluator.evaluate(self)
                # test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output]
                test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output if
                                  k.startswith('NDCG')]

                # # used to draw graph for 5 groups of user changes
                # s_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'mainstream_scores')
                # s_file = os.path.join(s_dir, 'MultVAE_scores_distribution_more_epoch')
                # if not os.path.exists(s_file):
                #     os.mkdir(s_file)
                # ndcg_test_all = evaluator.evaluate(self, mean=False)
                # with open(os.path.join(s_file, str(epoch) + '_epoch.npy'), 'wb') as f:
                #     np.save(f, ndcg_test_all)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        # torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))
                        # save scores for all users
                        # rec = self.predict_all()
                        best_result = test_score_output
                        # ndcg_test_all = evaluator.evaluate(self, mean=False)
                        # # similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name,
                        # #                               'mainstream_scores')
                        # similarity_file = os.path.join(similarity_dir, 'MultVAE_scores')
                        # if not os.path.exists(similarity_file):
                        #     os.mkdir(similarity_file)
                        # with open(os.path.join(similarity_file, self.time + '_vae_scores.npy'), 'wb') as f:
                        #     np.save(f, ndcg_test_all)

                        if self.anneal_cap == 1: print(self.anneal)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
                # epoch_info += ['ndcg@20= ' + str(ndcg[3])]
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        # return early_stop.best_score, total_train_time
        # return self.es.best_score, total_train_time
        return best_result, total_train_time

    def train_model_per_batch(self, batch_matrix, batch_weight=None):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        output, kl_loss = self.forward(batch_matrix)

        # loss
        # ce_loss = -(F.log_softmax(output, 1) * batch_matrix).mean()
        if batch_weight is None:
            ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()
        else:
            ce_loss = -((F.log_softmax(output, 1) * batch_matrix) * batch_weight.view(output.shape[0], -1)).sum(
                1).mean()
            # # sum of all
            # ce_loss = -((F.log_softmax(output, 1) * batch_matrix) * batch_weight.view(output.shape[0], -1)).sum()

        loss = ce_loss + kl_loss * self.anneal

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        self.update_count += 1

        return loss

    # def get_rec(self):
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

    def get_rec(self, eval_input):
        self.eval()
        with torch.no_grad():
            eval_output = self.forward(eval_input).detach().cpu().numpy()
        self.train()
        return eval_output

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            eval_output = self.forward(eval_input).detach().cpu().numpy()

            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output