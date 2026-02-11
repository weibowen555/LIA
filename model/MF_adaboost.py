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
# from concurrent.futures import ThreadPoolExecutor

# np.random.seed(0)
# torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="resource_tracker")


class MF_adaboost(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MF_adaboost, self).__init__(dataset, model_conf)
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

        s_file = os.path.join(similarity_dir, 'MF_adaboost_records')
        if not os.path.exists(s_file):
            os.mkdir(s_file)
        similarity_file = os.path.join(s_file, self.time + '_MF_record_scores')
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
        # self.stump_weights = [] ### design 1, here to change, comment out
        # self.sample_weights_list = [] ###
        # self.errors = [] ###
        # self.ada_errors = []
        self.user_err_means_list = []
        self.item_err_means_list = []

        # initializing weights uniformly
        # how to initialize sample weights and put it into MF
        # self.sample_weights[0] = np.ones(shape=m) / m
        # # self.sample_weights[0] = np.ones(shape=(m, n)) / n
        # self.sample_weights_users = np.ones(shape=(m, 1)) / m
        # self.sample_weights_items = np.ones(shape=(n, 1)) / n

        # self.sample_weights.append(np.ones(shape=(m, n)))
        # self.sample_weights.append(np.ones(shape=(m, n)) / (m * n)) # desgin one

        self.sample_weights = np.ones(shape=(m, n)) / (m * n)
        # local_stump = MF(dataset, self.model_conf, self.device)

        for t in tqdm(range(self.iters)):
            # fitting weak learner
            es = copy.deepcopy(early_stop)
            # curr_sample_weights = self.sample_weights[t]

            curr_sample_weights = self.sample_weights
            # cur_sample_users = self.sample_weights_users
            # cur_sample_items = self.sample_weights_items
            stump = MF(dataset, self.model_conf, self.device)
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


            ###
            user_err_list, item_err_list = [], []
            for i in range(self.neg_sample_num):
                # temp_users = [[] for _ in range(self.num_users)]
                # temp_items = [[] for _ in range(self.num_items)]
                user_list, item_list, label_list = MP_Utility.negative_sampling(self.num_users,
                                                                                self.num_items,
                                                                                self.train_df[0],
                                                                                self.train_df[1],
                                                                                self.neg_sample_rate_eval)

                # pos_user, neg_user, pos_item, neg_item, pos_label, neg_label = MP_Utility.negative_sampling_full(self.num_users,
                #                                                                 self.num_items,
                #                                                                 self.train_df[0],
                #                                                                 self.train_df[1],
                #                                                                 self.neg_sample_rate_eval)

                # user_list, item_list, label_list = MP_Utility.positive_sampling(self.num_users,
                #                                                                 self.num_items,
                #                                                                 self.train_df[0],
                #                                                                 self.train_df[1],
                #                                                                 )

                u_emd, v_emd = stump.user_factors.weight.detach().cpu().numpy(), stump.item_factors.weight.detach().cpu().numpy()

                u, v = u_emd[user_list], v_emd[item_list]

                # u_pos, v_pos = u_emd[pos_user], v_emd[pos_item]
                # u_neg, v_neg = u_emd[neg_user], v_emd[neg_item]

                # rec_list = (stump.user_factors(user_list).detach().cpu().numpy() * stump.item_factors(item_list).detach().cpu().numpy()).sum(axis=1)
                # u_list, i_list = torch.Tensor(user_list).int().to(self.device), torch.Tensor(item_list).int().to(self.device)
                # rec_list = ((stump.user_factors(u_list) * stump.item_factors(i_list)).sum(1)).detach().cpu().numpy()

                # binary cross entropy
                rec_list = (u * v).sum(axis=1)

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
            # self.user_err_means_list.append(user_err_mean)
            # self.item_err_means_list.append(item_err_mean)

            # save
            self.u_file = os.path.join(similarity_file, 'user_vectors')
            if not os.path.exists(self.u_file):
                os.mkdir(self.u_file)

            self.i_file = os.path.join(similarity_file, 'item_vectors')
            if not os.path.exists(self.i_file):
                os.mkdir(self.i_file)

            with open(os.path.join(self.u_file, str(t) +'_u.npy'), 'wb') as f:
                np.save(f, user_err_mean)

            with open(os.path.join(self.i_file, str(t) +'_i.npy'), 'wb') as f:
                np.save(f, item_err_mean)


            err = np.matmul(user_err_mean, item_err_mean.T) # user * items, full version

            # err = user_err_mean
            # err = item_err_mean

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

            # # for item
            # new_sample_weights = curr_sample_weights.T * np.exp(stump_weight * err)

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

            # w_dir = os.path.join(similarity_dir, 'selcted_users_items')
            # if not os.path.exists(w_dir):
            #     os.mkdir(w_dir)
            # m_user0 = os.path.join(w_dir, 'u_main_0')
            # m_user1 = os.path.join(w_dir, 'u_main_1')
            # m_user2 = os.path.join(w_dir, 'u_main_2')
            # m_user3 = os.path.join(w_dir, 'u_main_3')
            # a_user0 = os.path.join(w_dir, 'a_user0')
            # a_user1 = os.path.join(w_dir, 'a_user1')
            # a_user2 = os.path.join(w_dir, 'a_user2')
            # a_user3 = os.path.join(w_dir, 'a_user3')
            # m_item0 = os.path.join(w_dir, 'm_item0')
            # m_item1 = os.path.join(w_dir, 'm_item1')
            # m_item2 = os.path.join(w_dir, 'm_item2')
            # m_item3 = os.path.join(w_dir, 'm_item3')
            # p_item0 = os.path.join(w_dir, 'p_item0')
            # p_item0 = os.path.join(w_dir, 'p_item1')
            # p_item0 = os.path.join(w_dir, 'p_item2')
            # p_item0 = os.path.join(w_dir, 'p_item3')
            #
            # if not os.path.exists(m_user0):
            #     os.mkdir(m_user0)
            # stump_w_alpha0 = os.path.join(m_user0, self.time + '_weights_alpha')
            # if not os.path.exists(stump_w_alpha0):
            #     os.mkdir(stump_w_alpha0)
            # with open(os.path.join(stump_w_alpha0, str(t) + '_alpha.npy'), 'wb') as f:
            #     np.save(f, stump_weight[7154, :])
            # err0 = os.path.join(m_user0, self.time + '_err')
            # if not os.path.exists(err0):
            #     os.mkdir(err0)
            # with open(os.path.join(err0, str(t) + '_err.npy'), 'wb') as f:
            #     np.save(f, err[7154, :])
            # sample_w0 = os.path.join(m_user0, self.time + '_sample_w')
            # if not os.path.exists(sample_w0):
            #     os.mkdir(sample_w0)
            # with open(os.path.join(sample_w0, str(t) + '_sample_w.npy'), 'wb') as f:
            #     np.save(f, curr_sample_weights[7154, :])
            #
            # # m_user1 = os.path.join(w_dir, 'u_main_1')
            # if not os.path.exists(m_user1):
            #     os.mkdir(m_user1)
            # stump_w_alpha1 = os.path.join(m_user1, self.time + '_weights_alpha')
            # if not os.path.exists(stump_w_alpha1):
            #     os.mkdir(stump_w_alpha1)
            # with open(os.path.join(stump_w_alpha1, str(t) + '_alpha.npy'), 'wb') as f:
            #     np.save(f, stump_weight[11630, :])
            # err1 = os.path.join(m_user1, self.time + '_err')
            # if not os.path.exists(err1):
            #     os.mkdir(err1)
            # with open(os.path.join(err1, str(t) + '_err.npy'), 'wb') as f:
            #     np.save(f, err[11630, :])
            # sample_w1 = os.path.join(m_user1, self.time + '_sample_w')
            # if not os.path.exists(sample_w1):
            #     os.mkdir(sample_w1)
            # with open(os.path.join(sample_w1, str(t) + '_sample_w.npy'), 'wb') as f:
            #     np.save(f, curr_sample_weights[11630, :])
            #
            # # m_user2 = os.path.join(w_dir, 'u_main_2')
            # if not os.path.exists(m_user2):
            #     os.mkdir(m_user2)
            # stump_w_alpha2 = os.path.join(m_user2, self.time + '_weights_alpha')
            # if not os.path.exists(stump_w_alpha2):
            #     os.mkdir(stump_w_alpha2)
            # with open(os.path.join(stump_w_alpha2, str(t) + '_alpha.npy'), 'wb') as f:
            #     np.save(f, stump_weight[6354, :])
            # err2 = os.path.join(m_user2, self.time + '_err')
            # if not os.path.exists(err2):
            #     os.mkdir(err2)
            # with open(os.path.join(err2, str(t) + '_err.npy'), 'wb') as f:
            #     np.save(f, err[6354, :])
            # sample_w2 = os.path.join(m_user2, self.time + '_sample_w')
            # if not os.path.exists(sample_w2):
            #     os.mkdir(sample_w2)
            # with open(os.path.join(sample_w2, str(t) + '_sample_w.npy'), 'wb') as f:
            #     np.save(f, curr_sample_weights[6354, :])
            #
            # if not os.path.exists(m_user3):
            #     os.mkdir(m_user3)
            # stump_w_alpha3 = os.path.join(m_user3, self.time + '_weights_alpha')
            # if not os.path.exists(stump_w_alpha3):
            #     os.mkdir(stump_w_alpha3)
            # with open(os.path.join(stump_w_alpha3, str(t) + '_alpha.npy'), 'wb') as f:
            #     np.save(f, stump_weight[5325, :])
            # err3 = os.path.join(m_user3, self.time + '_err')
            # if not os.path.exists(err3):
            #     os.mkdir(err3)
            # with open(os.path.join(err3, str(t) + '_err.npy'), 'wb') as f:
            #     np.save(f, err[5325, :])
            # sample_w3 = os.path.join(m_user3, self.time + '_sample_w')
            # if not os.path.exists(sample_w3):
            #     os.mkdir(sample_w3)
            # with open(os.path.join(sample_w3, str(t) + '_sample_w.npy'), 'wb') as f:
            #     np.save(f, curr_sample_weights[5325, :])
            #
            # if not os.path.exists(a_user0):
            #     os.mkdir(a_user0)
            # stump_w_alpha0 = os.path.join(a_user0, self.time + '_weights_alpha')
            # if not os.path.exists(stump_w_alpha0):
            #     os.mkdir(stump_w_alpha0)
            # with open(os.path.join(stump_w_alpha0, str(t) + '_alpha.npy'), 'wb') as f:
            #     np.save(f, stump_weight[1298, :])
            # err0 = os.path.join(a_user0, self.time + '_err')
            # if not os.path.exists(err0):
            #     os.mkdir(err0)
            # with open(os.path.join(err0, str(t) + '_err.npy'), 'wb') as f:
            #     np.save(f, err[1298, :])
            # sample_w0 = os.path.join(a_user1, self.time + '_sample_w')
            # if not os.path.exists(sample_w0):
            #     os.mkdir(sample_w0)
            # with open(os.path.join(sample_w0, str(t) + '_sample_w.npy'), 'wb') as f:
            #     np.save(f, curr_sample_weights[1298, :])




            self.sample_weights = new_sample_weights

            # self.sample_weights = new_sample_weights.T  # for item

            # self.stumps[t] = stump
            # self.stump_weights[t] = stump_weight
            # self.errors[t] = err
            # self.ada_errors[t] = np.prod(((self.errors[t] * (1 - self.errors[t])) ** 1 / 2))
            self.stumps.append(stump)

            # self.stump_weights.append(stump_weight) ## memory consuming ###### design 1, here to change

            # ###
            # self.sample_weights_list.append(curr_sample_weights) ###
            ### her

            # # self.errors.append(err) ###
            # w_dir = os.path.join(similarity_dir, 'save_infos')
            # if not os.path.exists(w_dir):
            #     os.mkdir(w_dir)
            # w_file = os.path.join(w_dir, self.time + '_weights_err_saves')
            # if not os.path.exists(w_file):
            #     os.mkdir(w_file)
            # with open(os.path.join(w_file, str(t) + '_err.npy'), 'wb') as f:
            #     np.save(f, err)

            # torch.cuda.empty_cache()
            # self.ada_errors.append(np.prod(((self.errors[t] * (1 - self.errors[t])) ** 1 / 2)))

        # test_score_output = evaluator.evaluate(self)

        # test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output]
        # test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output if k.startswith('NDCG')]

        # ndcg_test_all = evaluator.evaluate(self, mean=False)

        test_score_output, ndcg_test_all = evaluator.evaluate_full_boost(self)
        mf_boost_file = os.path.join(similarity_dir, 'MF_adaboost_scores')
        if not os.path.exists(mf_boost_file):
            os.mkdir(mf_boost_file)
        with open(os.path.join(mf_boost_file, self.time + '_boost_scores.npy'), 'wb') as f:
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

        # # save weights and err, ### memeory!!!
        # w_dir = os.path.join(similarity_dir, 'save_infos')
        # if not os.path.exists(w_dir):
        #     os.mkdir(w_dir)
        # w_file = os.path.join(w_dir, self.time + '_weights_err_saves')
        # if not os.path.exists(w_file):
        #     os.mkdir(w_file)
        # with open(os.path.join(w_file, 'err.npy'), 'wb') as f:
        #     np.save(f, self.errors)
        # # err can used to calculate alphas and sample weights
        # # with open(os.path.join(w_file, 'alpha.npy'), 'wb') as f:
        # #     np.save(f, self.stump_weights[5])
        # # with open(os.path.join(w_file, 'sample_weights.npy'), 'wb') as f:
        # #     np.save(f, self.sample_weights)

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
        # # design 1, here to change to comment out
        # rec = None
        # stump_weights = torch.tensor(self.stump_weights, dtype=torch.float32)
        # stump_weights = F.softmax(stump_weights / self.tau) # design 1
        # for i, stump in enumerate(self.stumps):
        #     if i == 0:
        #         rec = stump_weights[i] * stump.get_rec_tensor()
        #     else:
        #         rec += stump_weights[i] * stump.get_rec_tensor()
        # return rec.detach().cpu().numpy()

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
            # err = np.matmul(self.user_err_means_list[i], self.item_err_means_list[i].T)

            # load save files
            users = np.load(self.u_file + '/' + str(i) + '_u.npy', allow_pickle=True)
            items = np.load(self.i_file + '/' + str(i) + '_i.npy', allow_pickle=True)
            # err = np.matmul(users, items.T)
            # stump_weight = np.log((self.beta1) / (self.beta2 + err))
            # alpha = softmax(stump_weight / self.tau)

            # full version
            err = torch.matmul(torch.Tensor(users).float().to(self.device),
                               torch.Tensor(items.T).float().to(self.device))

            # err = torch.Tensor(users).float().to(self.device) # user
            # err = torch.Tensor(items.T).float().to(self.device) # item

            stump_weight = torch.log((self.beta1) / (self.beta2 + err))
            alpha = F.softmax(stump_weight / self.tau)
            if i == 0:
                rec = alpha * stump.get_rec_tensor()
            else:
                rec += alpha * stump.get_rec_tensor()

        return rec.detach().cpu().numpy()

        # num_threads = 5
        #
        # with ThreadPoolExecutor(max_workers=num_threads) as executor:
        #     tasks = [(i, stump, self.stump_weights[i], self.tau) for i, stump in enumerate(self.stumps)]
        #     results = list(executor.map(self.parallelized_task, tasks))
        #
        # return np.sum(results, axis=0)

class MF(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MF, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.display_step = model_conf['display_step']
        self.hidden_neuron = model_conf['emb_dim']
        self.neg_sample_rate = model_conf['neg_sample_rate']

        self.batch_size = model_conf['batch_size']
        self.regularization = model_conf['reg']
        self.lr = model_conf['lr']
        self.train_df = dataset.train_df
        self.device = device
        self.loss_function = torch.nn.MSELoss()
        # self.train_like = dataset.train_like
        # self.test_like = dataset.test_like
        # self.user_list, self.item_list, self.label_list = MP_Utility.negative_sampling(self.num_users, self.num_items,
        #                                                                             self.train_df[0],
        #                                                                             self.train_df[1],
        #                                                                             self.neg_sample_rate)
        print('******************** MF ********************')
        self.user_factors = torch.nn.Embedding(self.num_users, self.hidden_neuron)  # , sparse=True
        # self.user_factors.weight.data.uniform_(-0.05, 0.05)
        self.item_factors = torch.nn.Embedding(self.num_items, self.hidden_neuron)  # , sparse=True
        # self.item_factors.weight.data.uniform_(-0.05, 0.05)
        nn.init.xavier_normal_(self.user_factors.weight)
        nn.init.xavier_normal_(self.item_factors.weight)
        print('P: ', self.user_factors)
        print('Q: ', self.item_factors)
        self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() + LA.norm(self.item_factors.weight.data, 'fro').item())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.regularization)
        self.time = strftime('%Y%m%d-%H%M')

        print('********************* MF Initialization Done *********************')
        self.to(self.device)

    def forward(self, user, item):
        # Get the dot product per row
        u = self.user_factors(user)
        v = self.item_factors(item)
        x = (u * v).sum(1)
        return x

    def train_model(self, dataset, evaluator, early_stop, logger, config, similarity_file, sample_weights):
        exp_config = config['Experiment']
        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir
        # users = np.arange(self.num_users)
        # similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
        # if not os.path.exists(similarity_dir):
        #     os.mkdir(similarity_dir)

        # torch.autograd.set_detect_anomaly(True)

        best_result = None

        start = time()
        for epoch_itr in range(1, num_epochs + 1):
            self.train()
            ndcg_test_all = None
            epoch_cost = 0.
            # epoch_cost1 = 0.
            # epoch_cost2 = 0.
            # self.user_list, self.item_list, self.label_list = MP_Utility.negative_sampling(self.num_users,
            #                                                                                self.num_items,
            #                                                                                self.train_df[0],
            #                                                                                self.train_df[1],
            #                                                                                self.neg_sample_rate)
            self.user_list, self.item_list, self.label_list, self.weights = MP_Utility.negative_sampling_boost(
                self.num_users,
                self.num_items,
                self.train_df[0],
                self.train_df[1],
                self.neg_sample_rate,
                sample_weights)
            # start_time = time() * 1000.0
            batch_loader = DataBatcher(np.arange(len(self.user_list)), batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                # tmp_cost, tmp_cost1, tmp_cost2 = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
                #                                                   self.label_list[batch_idx])
                # users, items = self.user_list[batch_idx], self.item_list[batch_idx]
                tmp_cost = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
                                            self.label_list[batch_idx], self.weights[batch_idx])
                epoch_cost += tmp_cost
                # epoch_cost1 += tmp_cost1
                # epoch_cost2 += tmp_cost2
                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, tmp_cost))
            epoch_train_time = time() - epoch_train_start
            epoch_info = ['epoch=%3d' % epoch_itr, 'loss=%.3f' % epoch_cost, 'train time=%.2f' % epoch_train_time]

            # self.train_model_help(epoch_itr)

            ## evaluation
            if (epoch_itr >= test_from and epoch_itr % test_step == 0) or epoch_itr == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                # test_score = evaluator.evaluate(self)
                test_score = evaluator.evaluate_vali(self)
                # test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]
                #
                # print(test_score_str)
                updated, should_stop = early_stop.step(test_score, epoch_itr)
                test_score_output = evaluator.evaluate(self)
                test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output if k.startswith('NDCG')]
                # test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output]

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        # torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))
                        # if self.anneal_cap == 1: print(self.anneal)
                        best_result = test_score_output
                        # ndcg_test_all = evaluator.evaluate(self, mean=False)

                        # print(ndcg_test_all['DG@20'])
                        # print(mean(ndcg_test_all['DG@20']))
                        # similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name,
                        #                               'mainstream_scores')
                        # self.make_records()

                        # # save
                        # with open(os.path.join(similarity_file, self.time + '_mf_scores.npy'), 'wb') as f:
                        #     np.save(f, ndcg_test_all)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch_itr % print_step == 0:
                logger.info(', '.join(epoch_info))

            # total_train_time = time() - start

        # torch.autograd.set_detect_anomaly(False)

        return best_result, time() - start

    def train_batch(self, user_input, item_input, label_input, weights):
        # reset gradients
        self.optimizer.zero_grad()
        users = torch.Tensor(user_input).int().to(self.device)
        items = torch.Tensor(item_input).int().to(self.device)
        labels = torch.Tensor(label_input).float().to(self.device)
        weights = torch.Tensor(weights).float().to(self.device)
        # labels = torch.Tensor(label_input).double().to(self.device)
        # weights = torch.Tensor(weights).double().to(self.device)
        # i_w = torch.Tensor(i_w).float().to(self.device)
        total_loss = 0

        # regularization term change?
        self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() + LA.norm(self.item_factors.weight.data, 'fro').item())
        # y_hat = self.forward(users, items, u_w, i_w)
        y_hat = self.forward(users, items)
        # y_hat_sig = F.sigmoid(y_hat)
        # loss = F.mse_loss(y_hat, labels)   ## check this
        # loss = self.loss_function(y_hat, labels)
        # added_loss = loss.item() + self.regularization_term

        # # MSE
        # loss = (weights * (y_hat - labels) ** 2).mean()    ### check this y_sig_hat
        # added_loss = loss + self.regularization_term

        # binary cross entropy loss
        y_hat_sig = F.sigmoid(y_hat)
        loss = -(weights * (labels * torch.log(y_hat_sig) + (1 - labels) * torch.log(1 - y_hat_sig))).sum() # or mean, full version

        # ablation
        # user
        # loss = -(torch.matmul(weights, (labels * torch.log(y_hat_sig) + (1 - labels) * torch.log(1 - y_hat_sig)))).sum()
        # # item
        # loss = -(torch.matmul((labels * torch.log(y_hat_sig) + (1 - labels) * torch.log(1 - y_hat_sig)), weights.T)).sum()

        added_loss = loss + self.regularization_term

        total_loss += added_loss
        # backpropagate
        loss.backward()
        # update
        self.optimizer.step()

        # self.eval()

        # return (total_loss, total_loss1, total_loss2)
        return total_loss


    def get_rec_tensor(self):
        P, Q = self.user_factors.weight, self.item_factors.weight
        Rec = torch.matmul(P, Q.T)
        return Rec
    def get_rec(self):
        P, Q = self.user_factors.weight, self.item_factors.weight
        P = P.detach().cpu().numpy()
        Q = Q.detach().cpu().numpy()
        Rec = np.matmul(P, Q.T)
        return Rec

    def make_records(self):  # record all the results' details into files
        P, Q = self.user_factors.weight, self.item_factors.weight
        P = P.detach().cpu().numpy()
        Q = Q.detach().cpu().numpy()
        similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
        if not os.path.exists(similarity_dir):
            os.mkdir(similarity_dir)
        similarity_file = os.path.join(similarity_dir, 'PC_saves')
        if not os.path.exists(similarity_file):
            os.mkdir(similarity_file)
        with open(os.path.join(similarity_file,'P_MF.npy'), 'wb') as f:
            np.save(f, P)
        with open(os.path.join(similarity_file,'Q_MF.npy'), 'wb') as f:
            np.save(f, Q)
        # return P, Q

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            # eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            # P = torch.Tensor(self.user_list[user_ids]).int()
            # Q = torch.Tensor(self.item_list[user_ids]).int()
            Rec = self.get_rec()
            eval_output = Rec[user_ids, :]
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output