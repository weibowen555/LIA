import os
import sys
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from utils import Tool
from utils import MP_Utility
from experiment import EarlyStop
from time import strftime

from collections import defaultdict

from rrl.components_rrl import BinarizeLayer
from rrl.components_rrl import UnionLayer, LearnableUnionLayer, LRLayer, UnionLayer_AND, UnionLayer_OR
from utils.GSAM import GSAM
import torch.distributed as dist
import re
import copy

# class Net(BaseRecommender):
#     def __init__(self, dim_list, use_not=False, left=None, right=None, use_nlaf=False, estimated_grad=False, use_skip=False, alpha=0.999, beta=8, gamma=1, temperature=0.01):
#         super(Net, self).__init__()

#         self.dim_list = dim_list
#         self.use_not = use_not
#         self.left = left
#         self.right = right
#         self.layer_list = nn.ModuleList([])
#         self.use_skip = use_skip
#         self.t = nn.Parameter(torch.log(torch.tensor([temperature])))

#         prev_layer_dim = dim_list[0]
#         for i in range(1, len(dim_list)):
#             num = prev_layer_dim
            
#             skip_from_layer = None
#             if self.use_skip and i >= 4:
#                 skip_from_layer = self.layer_list[-2]
#                 num += skip_from_layer.output_dim

#             if i == 1:
#                 layer = BinarizeLayer(dim_list[i], num, self.use_not, self.left, self.right)
#                 layer_name = 'binary{}'.format(i)
#             elif i == len(dim_list) - 1:
#                 layer = LRLayer(dim_list[i], num)
#                 layer_name = 'lr{}'.format(i)
#             else:
#                 # The first logical layer does not use NOT if the binarization layer has already used NOT
#                 # layer_use_not = True if i != 2 else False
#                 layer_use_not = False
#                 layer = UnionLayer(dim_list[i], num, use_nlaf=use_nlaf, estimated_grad=estimated_grad, use_not=layer_use_not, alpha=alpha, beta=beta, gamma=gamma)
#                 layer_name = 'union{}'.format(i)
            
#             layer.conn = lambda: None  # create an empty class to save the connections
#             layer.conn.prev_layer = self.layer_list[-1] if len(self.layer_list) > 0 else None
#             layer.conn.is_skip_to_layer = False
#             layer.conn.skip_from_layer = skip_from_layer
#             if skip_from_layer is not None:
#                 skip_from_layer.conn.is_skip_to_layer = True

#             prev_layer_dim = layer.output_dim
#             self.add_module(layer_name, layer)
#             self.layer_list.append(layer)

#     def forward(self, x):
#         for layer in self.layer_list:
#             if layer.conn.skip_from_layer is not None:
#                 x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
#                 del layer.conn.skip_from_layer.x_res
#             x = layer(x)
#             if layer.conn.is_skip_to_layer:
#                 layer.x_res = x
#         return x
    
#     def bi_forward(self, x, count=False):
#         for layer in self.layer_list:
#             if layer.conn.skip_from_layer is not None:
#                 x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
#                 del layer.conn.skip_from_layer.x_res
#             x = layer.binarized_forward(x)
#             if layer.conn.is_skip_to_layer:
#                 layer.x_res = x
#             if count and layer.layer_type != 'linear':
#                 layer.node_activation_cnt += torch.sum(x, dim=0)
#                 layer.forward_tot += x.shape[0]
#         return x


class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    @property
    def layer_list(self):
        return self.module.layer_list
    
    @property
    def t(self):
        return self.module.t


class RRL(BaseRecommender):
    # def __init__(self, dim_list, device_id, use_not=False, is_rank0=False, log_file=None, writer=None, left=None,
    #              right=None, save_best=False, estimated_grad=False, save_path=None, distributed=True, use_skip=False, 
    #              use_nlaf=False, alpha=0.999, beta=8, gamma=1, temperature=0.01):
    #     super(RRL, self).__init__()

    def __init__(self, dataset, model_conf, device, is_rank0=True, distributed=False):
        super(RRL, self).__init__(dataset, model_conf)

        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.dropout = model_conf['dropout']
        self.reg = model_conf['reg']
        self.neg_sample_rate = model_conf['neg_sample_rate']

        self.train_df = dataset.train_df
        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']

        self.lr = model_conf['lr']

        self.device = device

        self.time = strftime('%Y%m%d-%H%M')

        # self.dim_list = dim_list # note: dim list should be [(m,n), 15, 32, 32, n]
        self.structure = model_conf['structure']
        self.dim_list = [self.num_items] + list(map(int, self.structure.split('@'))) + [self.num_items]
        self.lrdr = model_conf['lr_decay_rate']
        self.lrde = model_conf['lr_decay_epoch']

        self.use_not = model_conf['use_not']
        self.use_skip = model_conf['use_skip']
        self.use_nlaf = model_conf['use_nlaf']
        self.alpha =model_conf['alpha']
        self.beta = model_conf['beta']
        self.gamma = model_conf['gamma']

        self.is_rank0 = is_rank0
        self.estimated_grad = model_conf['estimated_grad']
        self.temperature = model_conf['temperature']
        self.distributed = distributed

        self.use_gsam = model_conf.get('use_gsam', False)
        self.gsam_rho = model_conf.get('gsam_rho', 0.05)
        self.gsam_alpha = model_conf.get('gsam_alpha', 0.1)
        self.use_learnable_gate = model_conf.get('use_learnable_gate', False)
        self.gate_init = model_conf.get('gate_init', 0.0)
        self.init_not = model_conf.get('init_not', False)
        self.use_input_skip = model_conf.get('use_input_skip', False)
        self.penalty_weight = model_conf.get('penalty_weight', 0.0)

        self.use_swa = model_conf.get('use_swa', False)
        self.swa_start = model_conf.get('swa_start', 50)
        self.swa_n = 0
        self.swa_state = None

        self.denoise_rate = model_conf.get('denoise_rate', 0.0)

        # if self.is_rank0:
        #     for handler in logging.root.handlers[:]:
        #         logging.root.removeHandler(handler)

        #     log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
        #     if log_file is None:
        #         logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
        #     else:
        #         logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)
        # self.writer = writer

        # self.net = Net(self.dim_list, use_not=self.use_not, left=None, right=None, use_nlaf=self.use_nlaf, estimated_grad=self.estimated_grad, use_skip=self.use_skip, alpha=self.alpha, beta=self.beta, gamma=self.gamma, temperature=self.temperature)
        # self.net.cuda(self.device)

        self.build()

        # loss, cross entropy loss
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        # self.criterion = nn.modules.loss.FocalLoss()

        # optimizer
        if self.use_gsam:
            self.optimizer = GSAM(self.parameters(), torch.optim.Adam, rho=self.gsam_rho, alpha=self.gsam_alpha, lr=self.lr, weight_decay=self.reg)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device)

        if self.distributed:
            MyDistributedDataParallel(self, device_ids=[self.device])

    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for layer in self.layer_list[: -1]:
            layer.clip()
    
    def edge_penalty(self):
        edge_penalty = 0.0
        for layer in self.layer_list[1: -1]:
            edge_penalty += layer.edge_count()
        return edge_penalty
    
    def l1_penalty(self):
        l1_penalty = 0.0
        for layer in self.layer_list[1: ]:
            l1_penalty += layer.l1_norm()
        return l1_penalty
    
    def l2_penalty(self):
        l2_penalty = 0.0
        for layer in self.layer_list[1: ]:
            l2_penalty += layer.l2_norm()
        return l2_penalty
    
    def mixed_penalty(self):
        penalty = 0.0
        for layer in self.layer_list[1: -1]:
            penalty += layer.l2_norm()
        penalty += self.layer_list[-1].l1_norm()
        return penalty

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def build(self):
        self.left, self.right = None, None
        dim_list = self.dim_list
        self.layer_list = nn.ModuleList([])
        self.t = nn.Parameter(torch.log(torch.tensor([self.temperature])))
        self.drop = nn.Dropout(p=self.dropout)

        prev_layer_dim = dim_list[0]
        for i in range(1, len(dim_list)):
            num = prev_layer_dim
            
            skip_from_layer = None
            if self.use_skip and i >= 4:
                skip_from_layer = self.layer_list[-2]
                num += skip_from_layer.output_dim
            if self.use_input_skip and i == len(dim_list) - 1:
                skip_from_layer = self.layer_list[0]  # BinarizeLayer
                num += skip_from_layer.output_dim

            if i == 1:
                layer = BinarizeLayer(dim_list[i], num, self.use_not, self.left, self.right)
                layer_name = 'binary{}'.format(i)
            elif i == len(dim_list) - 1:
                layer = LRLayer(dim_list[i], num)
                layer_name = 'lr{}'.format(i)

            # elif i == 2:
            #     layer_use_not = False
            #     layer = UnionLayer_AND(dim_list[i], num, use_nlaf=self.use_nlaf, estimated_grad=self.estimated_grad, use_not=layer_use_not, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            #     layer_name = 'union_and{}'.format(i)

            #     # layer_use_not = False
            #     # layer = UnionLayer_OR(dim_list[i], num, use_nlaf=self.use_nlaf, estimated_grad=self.estimated_grad, use_not=layer_use_not, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            #     # layer_name = 'union_or{}'.format(i)

            # elif i == 3:
            #     # layer_use_not = False
            #     # layer = UnionLayer_AND(dim_list[i], num, use_nlaf=self.use_nlaf, estimated_grad=self.estimated_grad, use_not=layer_use_not, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            #     # layer_name = 'union_and{}'.format(i)

            #     layer_use_not = False
            #     layer = UnionLayer_OR(dim_list[i], num, use_nlaf=self.use_nlaf, estimated_grad=self.estimated_grad, use_not=layer_use_not, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            #     layer_name = 'union_or{}'.format(i)
            
            ### original union-layer
            else:
                # The first logical layer does not use NOT if the binarization layer has already used NOT
                # layer_use_not = True if i != 2 else False
                layer_use_not = self.use_not
                if self.use_learnable_gate:
                    layer = LearnableUnionLayer(dim_list[i], num, use_nlaf=self.use_nlaf, estimated_grad=self.estimated_grad, use_not=layer_use_not, alpha=self.alpha, beta=self.beta, gamma=self.gamma, gate_init=self.gate_init, init_not=self.init_not)
                else:
                    layer = UnionLayer(dim_list[i], num, use_nlaf=self.use_nlaf, estimated_grad=self.estimated_grad, use_not=layer_use_not, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
                layer_name = 'union{}'.format(i)
            
            layer.conn = lambda: None  # create an empty class to save the connections
            layer.conn.prev_layer = self.layer_list[-1] if len(self.layer_list) > 0 else None
            layer.conn.is_skip_to_layer = False
            layer.conn.skip_from_layer = skip_from_layer
            if skip_from_layer is not None:
                skip_from_layer.conn.is_skip_to_layer = True

            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)
        
    def forward(self, x):
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                del layer.conn.skip_from_layer.x_res
            x = layer(x)
            if layer.conn.is_skip_to_layer:
                layer.x_res = x
            if layer.layer_type == 'union':
                x = self.drop(x)
        return x
    
    def bi_forward(self, x, count=False):
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                del layer.conn.skip_from_layer.x_res
            x = layer.binarized_forward(x)
            if layer.conn.is_skip_to_layer:
                layer.x_res = x
            if count and layer.layer_type != 'linear':
                layer.node_activation_cnt += torch.sum(x, dim=0)
                layer.forward_tot += x.shape[0]
        return x
    

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        if self.is_rank0:
            log_dir = logger.log_dir

        # prepare dataset
        # dataset.set_eval_data('valid')
        users = np.arange(self.num_users)

        train_matrix = dataset.train_matrix.toarray()
        train_matrix = torch.FloatTensor(train_matrix)
        best_result = None

        # zeros_mat = torch.zeros_like(train_matrix)

        # best_epoch = -1
        
        # ######
        # e_start = False

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            # if epoch - best_epoch > 10:
            #     break
            # self.criterion = nn.CrossEntropyLoss().to(self.device)

            self.optimizer = self.exp_lr_scheduler(self.optimizer, epoch, init_lr=self.lr, lr_decay_rate=self.lrdr,
                                              lr_decay_epoch=self.lrde)

            self.train()

            epoch_loss = 0.0
            self.abs_gradient_max = 0.0
            self.abs_gradient_avg = 0.0

            # self.user_list, self.item_list, self.label_list = MP_Utility.negative_sampling(self.num_users,
            #                                                                                self.num_items,
            #                                                                                self.train_df[0],
            #                                                                                self.train_df[1],
            #                                                                                self.neg_sample_rate)

            # mask = MP_Utility.negative_sampling_neu(self.num_users, self.num_items, self.train_df[0], self.train_df[1], dataset.train_matrix.toarray(), self.neg_sample_rate)
            # mask = torch.FloatTensor(mask)

            if self.distributed:
                batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=False)
            else:
                batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)

            # batch_loader = DataBatcher(np.arange(len(self.user_list)), batch_size=self.batch_size, drop_remain=False,
            #                            shuffle=True)
            # num_batches = len(batch_loader)

            # ======================== Train
            epoch_train_start = time()

            # # neg_sample implement ###
            # mask = torch.zeros_like(train_matrix)
            # mask[self.user_list, self.item_list] = torch.FloatTensor(self.label_list)

            # if epoch == self.epoch_start: ######
            #     e_start = True

            # if epoch < 10:
            #     rule_mat = torch.ones_like(train_matrix)
            # else:
            #     rule_mat = self.rule_mat(zeros_mat, train_loader=train_matrix)
            #     # rule_mat = torch.softmax(rule_mat, dim=1) * self.num_items
            #     rule_mat = F.softmax(rule_mat.sum(1)) * self.num_users

            for b, batch_idx in enumerate(batch_loader):
                # tmp_cost = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
                #                             self.label_list[batch_idx])

                # batch_matrix = train_matrix[self.user_list[batch_idx], :].to(self.device)
                # batch_matrix = mask[self.user_list[batch_idx], :].to(self.device) #

                batch_matrix = train_matrix[batch_idx].to(self.device)
                # batch_weight = rule_mat[batch_idx].to(self.device)
                # batch_mask = mask[batch_idx].to(self.device)

                ######
                # batch_loss = self.train_model_per_batch(batch_matrix, self.item_list[batch_idx], e_start)
                # batch_loss = self.train_model_per_batch(batch_matrix, batch_weight)
                batch_loss = self.train_model_per_batch(batch_matrix)

                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))

            if self.distributed:
                dist.barrier()

            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]
            similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
            if not os.path.exists(similarity_dir):
                os.mkdir(similarity_dir)

            # print(self.abs_gradient_max, self.abs_gradient_avg)
            gradient_info = ['abs_gradient_max=%.4f' % self.abs_gradient_max, 'abs_gradient_avg=%.4f' % self.abs_gradient_avg]

            # ======================== Evaluate
            if ((epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs) and self.is_rank0:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                test_score = evaluator.evaluate_vali(self)
                if self.distributed:
                    try:
                        ndcg1 = torch.tensor(test_score['NDCG@1']).to(self.device)
                        ndcg5 = torch.tensor(test_score['NDCG@5']).to(self.device)
                        ndcg10 = torch.tensor(test_score['NDCG@10']).to(self.device)
                        ndcg20 = torch.tensor(test_score['NDCG@20']).to(self.device)
                        torch.distributed.all_reduce(ndcg1, op=dist.ReduceOp.AVG)
                        torch.distributed.all_reduce(ndcg5, op=dist.ReduceOp.AVG)
                        torch.distributed.all_reduce(ndcg10, op=dist.ReduceOp.AVG)
                        torch.distributed.all_reduce(ndcg20, op=dist.ReduceOp.AVG)
                        test_score['NDCG@1'] = ndcg1.detach().cpu().numpy()
                        test_score['NDCG@5'] = ndcg5.detach().cpu().numpy()
                        test_score['NDCG@10'] = ndcg10.detach().cpu().numpy()
                        test_score['NDCG@20'] = ndcg20.detach().cpu().numpy()
                    except RuntimeError as e:
                        print(f"An error occurred during collective operation: {e}")

                updated, should_stop = early_stop.step(test_score, epoch)

                # test_score_output = evaluator.evaluate(self)

                test_score_output, ndcg_test_all = evaluator.evaluate_full_boost(self)
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
                        torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))
                        # save scores for all users
                        # rec = self.predict_all()
                        best_result = test_score_output

                        # ndcg_test_all = evaluator.evaluate(self, mean=False)

                        # similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name,
                        #                               'mainstream_scores')


                        ### save all rec here
                        # similarity_file = os.path.join(similarity_dir, 'rrl_scores')
                        # if not os.path.exists(similarity_file):
                        #     os.mkdir(similarity_file)
                        # with open(os.path.join(similarity_file, self.time + '_rrl_scores.npy'), 'wb') as f:
                        #     np.save(f, ndcg_test_all)


                        # ## save all results
                        # output = self.predict(np.arange(self.num_users), dataset.train_matrix)
                        # similarity_file2 = os.path.join(similarity_dir, 'neu_full_mat_score')
                        # if not os.path.exists(similarity_file2):
                        #     os.mkdir(similarity_file2)
                        # with open(os.path.join(similarity_file2, self.time + '_neu_scores_b.npy'), 'wb') as f:
                        #     np.save(f, output)

                        # if self.anneal_cap == 1: print(self.anneal)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
                epoch_info += gradient_info
                # epoch_info += ['ndcg@20= ' + str(ndcg[3])]

            elif ((epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs) and not self.is_rank0 and self.distributed:
                self.eval()

                test_score = evaluator.evaluate_vali(self)
                try:
                    # print("all reduce in device %s", self.device)
                    ndcg1 = torch.tensor(test_score['NDCG@1']).to(self.device)
                    ndcg5 = torch.tensor(test_score['NDCG@5']).to(self.device)
                    ndcg10 = torch.tensor(test_score['NDCG@10']).to(self.device)
                    ndcg20 = torch.tensor(test_score['NDCG@20']).to(self.device)
                    torch.distributed.all_reduce(ndcg1, op=dist.ReduceOp.AVG)
                    torch.distributed.all_reduce(ndcg5, op=dist.ReduceOp.AVG)
                    torch.distributed.all_reduce(ndcg10, op=dist.ReduceOp.AVG)
                    torch.distributed.all_reduce(ndcg20, op=dist.ReduceOp.AVG)
                    test_score['NDCG@1'] = ndcg1.detach().cpu().numpy()
                    test_score['NDCG@5'] = ndcg5.detach().cpu().numpy()
                    test_score['NDCG@10'] = ndcg10.detach().cpu().numpy()
                    test_score['NDCG@20'] = ndcg20.detach().cpu().numpy()
                except RuntimeError as e:
                    print("in device %s", self.device)
                    print(f"An error occurred during collective operation: {e}")

                updated, should_stop = early_stop.step(test_score, epoch)

                if should_stop:
                    break

            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]


            # if epoch % print_step == 0:
            #     logger.info(', '.join(epoch_info))
            if epoch % print_step == 0 and self.is_rank0 and self.distributed:
                logger.info(', '.join(epoch_info))
                print(', '.join(epoch_info))
            elif epoch % print_step == 0 and self.is_rank0: # for one gpu only
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        return best_result, total_train_time

    def _compute_loss(self, batch_matrix, batch_weight=None):
        # Denoising: randomly zero out positive interactions in input, but keep original targets
        if self.training and self.denoise_rate > 0:
            mask = torch.bernoulli(torch.full_like(batch_matrix, 1 - self.denoise_rate))
            corrupted_input = batch_matrix * mask
        else:
            corrupted_input = batch_matrix

        y_pred = self.forward(corrupted_input)
        if batch_weight is None:
            loss = -(F.log_softmax(y_pred, 1) * batch_matrix).sum(1).mean()
        else:
            loss = -((F.log_softmax(y_pred, 1) * batch_matrix) * batch_weight.view(y_pred.shape[0], -1)).sum(1).mean()
        return loss

    def train_model_per_batch(self, batch_matrix, batch_weight=None):
        self.optimizer.zero_grad()

        loss = self._compute_loss(batch_matrix, batch_weight)
        loss.backward()

        if self.use_gsam:
            # GSAM two-step: first_step perturbs weights using gradient from original point
            self.optimizer.first_step()

            # Second forward-backward at perturbed weights
            self.optimizer.zero_grad()
            loss_perturbed = self._compute_loss(batch_matrix, batch_weight)
            loss_perturbed.backward()

            # second_step computes GSAM gradient, restores weights, and steps
            self.optimizer.second_step()
        else:
            self.optimizer.step()

        for i, param in enumerate(self.parameters()):
            if param.grad is not None:
                self.abs_gradient_max = max(self.abs_gradient_max, abs(torch.max(param.grad)))
                self.abs_gradient_avg = torch.sum(torch.abs(param.grad)) / (param.grad.numel())

        self.clip()

        return loss


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
    
    

    def detect_dead_node(self, data_loader=None):
        with torch.no_grad():
            for layer in self.layer_list[:-1]:
                layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=self.device)
                layer.forward_tot = 0

            batch_loader = DataBatcher(np.arange(self.num_users), batch_size=512, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            for b, batch_idx in enumerate(batch_loader):
                x = data_loader[batch_idx].to(self.device)
                self.bi_forward(x, count=True)
            # for x, y in data_loader:
            #     x_bar = x.cuda(self.device_id)
            #     self.net.bi_forward(x_bar, count=True)



    def rule_print(self, feature_name=None, label_name=None, train_loader=None, file=sys.stdout, mean=None, std=None, display=True):
        if self.layer_list[1] is None and train_loader is None:
            raise Exception("Need train_loader for the dead nodes detection.")

        if feature_name is None:
            feature_name = self.dataset.num_items
        
        # if label_name is None:


        if train_loader is None:
            train_loader = self.dataset.train_matrix.toarray()
            train_loader = torch.FloatTensor(train_loader)

        # detect dead nodes first
        if self.layer_list[1].node_activation_cnt is None:
            self.detect_dead_node(train_loader)

        # for Binarize Layer
        self.layer_list[0].get_bound_name(feature_name, mean, std)  # layer_list[0].rule_name == bound_name

        # for Union Layer
        for i in range(1, len(self.layer_list) - 1):
            layer = self.layer_list[i]
            layer.get_rules(layer.conn.prev_layer, layer.conn.skip_from_layer)
            skip_rule_name = None if layer.conn.skip_from_layer is None else layer.conn.skip_from_layer.rule_name
            wrap_prev_rule = False if i == 1 else True  # do not warp the bound_name
            layer.get_rule_description((skip_rule_name, layer.conn.prev_layer.rule_name), wrap=wrap_prev_rule)

        # for LR Layr
        layer = self.layer_list[-1]
        layer.get_rule2weights(layer.conn.prev_layer, layer.conn.skip_from_layer)

        if not display:
            return layer.rule2weights

        print('RID', end='\t', file=file)

        # for i, ln in enumerate(label_name):
        #     print('{}(b={:.4f})'.format(ln, layer.bl[i]), end='\t', file=file)

        print('Support\tRule', file=file)
        for rid, w in layer.rule2weights:
            print(rid, end='\t', file=file)

            # for li in range(len(label_name)):
            #     print('{:.4f}'.format(w[li]), end='\t', file=file)

            for li in range(self.dataset.num_items):
                print('{:.4f}'.format(w[li]), end='\t', file=file)

            now_layer = self.layer_list[-1 + rid[0]]
            print('{:.4f}'.format((now_layer.node_activation_cnt[layer.rid2dim[rid]] / now_layer.forward_tot).item()),
                  end='\t', file=file)
            print(now_layer.rule_name[rid[1]].count('item'), now_layer.rule_name[rid[1]], end='\n', file=file)
        print('#' * 60, file=file)
        return layer.rule2weights

    
    def rule_mat(self, rule_mat, feature_name=None, label_name=None, train_loader=None, file=sys.stdout, mean=None, std=None, display=True):
        if self.layer_list[1] is None and train_loader is None:
            raise Exception("Need train_loader for the dead nodes detection.")

        if feature_name is None:
            feature_name = self.dataset.num_items
        
        # if label_name is None:


        if train_loader is None:
            print("None train loader")
            train_loader = self.dataset.train_matrix.toarray()
            train_loader = torch.FloatTensor(train_loader)

        # detect dead nodes first
        if self.layer_list[1].node_activation_cnt is None:
            self.detect_dead_node(train_loader)

        # for Binarize Layer
        self.layer_list[0].get_bound_name(feature_name, mean, std)  # layer_list[0].rule_name == bound_name

        # for Union Layer
        for i in range(1, len(self.layer_list) - 1):
            layer = self.layer_list[i]
            layer.get_rules(layer.conn.prev_layer, layer.conn.skip_from_layer)
            skip_rule_name = None if layer.conn.skip_from_layer is None else layer.conn.skip_from_layer.rule_name
            wrap_prev_rule = False if i == 1 else True  # do not warp the bound_name
            layer.get_rule_description((skip_rule_name, layer.conn.prev_layer.rule_name), wrap=wrap_prev_rule)

        # for LR Layr
        layer = self.layer_list[-1]
        layer.get_rule2weights(layer.conn.prev_layer, layer.conn.skip_from_layer)

        if not display:
            return layer.rule2weights

        combine_rules = []
        for rid, w in layer.rule2weights:
            now_layer = self.layer_list[-1 + rid[0]]
            combine_rules.append(now_layer.rule_name[rid[1]])
        
        
        for items in combine_rules:
            matched_item_idx = [int(match) for match in re.findall(r'\d+', items)]
            rule_mat[:, matched_item_idx] = torch.where(train_loader[:, matched_item_idx] > 0, rule_mat[:, matched_item_idx] + 1, rule_mat[:, matched_item_idx])

        return rule_mat



    def rule_print_checking(self, feature_name=None, label_name=None, train_loader=None, file=sys.stdout, mean=None, std=None, display=True):
        if self.layer_list[1] is None and train_loader is None:
            raise Exception("Need train_loader for the dead nodes detection.")

        if feature_name is None:
            feature_name = self.dataset.num_items
        
        # if label_name is None:


        if train_loader is None:
            train_loader = self.dataset.train_matrix.toarray()
            train_loader = torch.FloatTensor(train_loader)

        # detect dead nodes first
        if self.layer_list[1].node_activation_cnt is None:
            self.detect_dead_node(train_loader)

        # for Binarize Layer
        self.layer_list[0].get_bound_name(feature_name, mean, std)  # layer_list[0].rule_name == bound_name

        # for Union Layer
        for i in range(1, len(self.layer_list) - 1):
            layer = self.layer_list[i]
            layer.get_rules(layer.conn.prev_layer, layer.conn.skip_from_layer)
            skip_rule_name = None if layer.conn.skip_from_layer is None else layer.conn.skip_from_layer.rule_name
            wrap_prev_rule = False if i == 1 else True  # do not warp the bound_name
            layer.get_rule_description((skip_rule_name, layer.conn.prev_layer.rule_name), wrap=wrap_prev_rule)

        # for LR Layr
        layer = self.layer_list[-1]
        layer.get_rule2weights(layer.conn.prev_layer, layer.conn.skip_from_layer)

        if not display:
            return layer.rule2weights

        print('RID', end='\t', file=file)

        # for i, ln in enumerate(label_name):
        #     print('{}(b={:.4f})'.format(ln, layer.bl[i]), end='\t', file=file)

        print('Support\tRule', file=file)
        for rid, w in layer.rule2weights[:20]:
            print(rid, end='\t', file=file)

            # for li in range(len(label_name)):
            #     print('{:.4f}'.format(w[li]), end='\t', file=file)

            for li in range(self.dataset.num_items):
                print('{:.4f}'.format(w[li]), end='\t', file=file)

            now_layer = self.layer_list[-1 + rid[0]]
            print('{:.4f}'.format((now_layer.node_activation_cnt[layer.rid2dim[rid]] / now_layer.forward_tot).item()),
                  end='\t', file=file)
            print(now_layer.rule_name[rid[1]].count('item'), now_layer.rule_name[rid[1]], end='\n', file=file)
        print('#' * 60, file=file)
        dis = []
        dis_or = []
        dis_and = []
        check = []
        combine_rules = []
        all_rules = []

        for rid, v in layer.rule2weights:
            now_layer = self.layer_list[-1 + rid[0]]
            if rid[0] == -1:
                # print(now_layer.rule_name[rid[1]].count('item'), now_layer.rule_name[rid[1]], end='\\n', file=file)
                combine_rules.append(now_layer.rule_name[rid[1]])
            if '|' in now_layer.rule_name[rid[1]]:
                dis_or.append(now_layer.rule_name[rid[1]].count('item'))
            if '&' in now_layer.rule_name[rid[1]]:
                dis_and.append(now_layer.rule_name[rid[1]].count('item'))
                check.append(now_layer.rule_name[rid[1]])
            dis.append(now_layer.rule_name[rid[1]].count('item'))
            all_rules.append(now_layer.rule_name[rid[1]])

        return layer.rule2weights, dis, dis_or, dis_and, check, combine_rules, all_rules