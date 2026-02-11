import os
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
from nfrl.components import BinarizeLayer
from nfrl.components import LRLayer, Selection_Layer, Selection_Layer_mask


# guess (more nodes, less items per rule?) gpu mem issue
# do some analysis.

# class NeuralRecommender(BaseRecommender):
#     def __init__(self, dataset, model_conf, device):
#         super(NeuralRecommender, self).__init__(dataset, model_conf)

class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    @property
    def layer_list(self):
        return self.module.layer_list

class NeuralRecommender(BaseRecommender):
    def __init__(self, dataset, model_conf, device, is_rank0=True, distributed=False):
        super(NeuralRecommender, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.dropout = model_conf['dropout']
        self.reg = model_conf['reg']

        self.neg_sample_rate = model_conf['neg_sample_rate']

        # self.train_df = dataset.train_df
        self.train_df = np.where(dataset.train_matrix.toarray() == 1)

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']

        self.lr = model_conf['lr']

        # self.epoch_start = model_conf['epoch_start'] ######

        self.device = device

        self.time = strftime('%Y%m%d-%H%M')

        # self.dim_list = dim_list # note: dim list should be [n, 15, 32, 32, n]
        self.structure = model_conf['structure']
        self.dim_list = [self.num_items] + list(map(int, self.structure.split('@'))) + [self.num_items]
        self.lrdr = model_conf['lr_decay_rate']
        self.lrde = model_conf['lr_decay_epoch']



        # self.build_graph()
        self.NFRL_NET(self.dim_list, left=None, right=None,)

        # loss, cross entropy loss
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        # self.criterion = nn.modules.loss.FocalLoss()

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device)

        print("model device: ", device)

        self.is_rank0 = is_rank0

        if distributed:
            print("distributed data parallel.")
            MyDistributedDataParallel(self, device_ids=[self.device])

    def NFRL_NET(self, dim_list, left=None, right=None,):
        # for NFRL
        self.layer_list = nn.ModuleList([])
        self.left, self.right = left, right
        prev_layer_dim = dim_list[0]

        # we do not need [1628,15,32,32,1628], we do not need 15, because all discrete
        for i in range(1, len(dim_list)):
            num = prev_layer_dim

            # ###
            # if i == 4: # last layer for 2 selection layers, actually skip layer combine
            #     num += self.layer_list[-2].output_dim

            # if i == 5: ## add one more, last layer for 3 selection layers
            #     num += (self.layer_list[-2].output_dim + self.layer_list[-3].output_dim)

            if i == 1:
                layer = BinarizeLayer(dim_list[i], num, self.left, self.right)
                layer_name = 'binary{}'.format(i)
            elif i == 2 and i != len(dim_list) - 1:
                # First selection layer (only if not the final layer)
                layer = Selection_Layer(dim_list[i], num)
                layer_name = 'selection{}'.format(i)
            elif i == 3 and i != len(dim_list) - 1:
                # Second selection layer (only if not the final layer)
                layer = Selection_Layer_mask(dim_list[i], num)
                layer_name = 'selection{}'.format(i)
            elif i == len(dim_list) - 1:
                # Final linear layer
                layer = LRLayer(dim_list[i], num)
                layer_name = 'lr{}'.format(i)

            # elif i == 4: ###
            #     layer = Selection_Layer_mask(dim_list[i], num)
            #     layer_name = 'selection{}'.format(i)

            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer) ###
            self.layer_list.append(layer)

    def forward(self, x):
        # nfrl here, check this
        x_res = None
        prev_w_op = None

        for i, layer in enumerate(self.layer_list):
            layer_type = getattr(layer, 'layer_type', None)

            if i == 0:
                # BinarizeLayer
                x = layer(x)
            elif layer_type == 'selection_layer':
                # First selection layer (returns x, prev_w_op)
                x, prev_w_op = layer(x)
                x_res = (x + 1) / 2
            elif layer_type == 'selection_mask':
                # Second selection layer (accepts prev_w_op)
                x, _ = layer(x, prev_w_op=prev_w_op)
                x = (x + 1) / 2  # make output in [0, 1]
            elif layer_type == 'linear':
                # Final LRLayer
                x = layer(x)
            else:
                # Default: just call forward
                x = layer(x)

        return x

    def clip(self):
        # add somewhere after optimizer.step()
        for layer in self.layer_list[: -1]:
            layer.clip()
    
    # ######
    # def clip_help(self):
    #     print('clip_help')
    #     for layer in self.layer_list[: -1]:
    #         layer.clip_help()


    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train_model(self, dataset, evaluator, early_stop, logger, config):
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
        
        # ######
        # e_start = False

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            # if epoch - best_epoch > 10:
            #     break
            # self.criterion = nn.CrossEntropyLoss().to(self.device)

            # self.optimizer = self.exp_lr_scheduler(self.optimizer, epoch, init_lr=self.lr, lr_decay_rate=self.lrdr,
            #                                   lr_decay_epoch=self.lrde)

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

            for b, batch_idx in enumerate(batch_loader):
                # tmp_cost = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
                #                             self.label_list[batch_idx])

                # batch_matrix = train_matrix[self.user_list[batch_idx], :].to(self.device)
                # batch_matrix = mask[self.user_list[batch_idx], :].to(self.device) #

                batch_matrix = train_matrix[batch_idx].to(self.device)
                # batch_mask = mask[batch_idx].to(self.device)

                ######
                # batch_loss = self.train_model_per_batch(batch_matrix, self.item_list[batch_idx], e_start)
                # batch_loss = self.train_model_per_batch(batch_matrix, batch_mask)
                batch_loss = self.train_model_per_batch(batch_matrix)

                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]
            similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
            if not os.path.exists(similarity_dir):
                os.mkdir(similarity_dir)

            # print(self.abs_gradient_max, self.abs_gradient_avg)
            gradient_info = ['abs_gradient_max=%.4f' % self.abs_gradient_max, 'abs_gradient_avg=%.4f' % self.abs_gradient_avg]
            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                test_score = evaluator.evaluate_vali(self)
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
                        similarity_file = os.path.join(similarity_dir, 'neu_scores')
                        if not os.path.exists(similarity_file):
                            os.mkdir(similarity_file)
                        with open(os.path.join(similarity_file, self.time + '_neu_scores.npy'), 'wb') as f:
                            np.save(f, ndcg_test_all)


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
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            # if epoch % print_step == 0:
            #     logger.info(', '.join(epoch_info))
            if epoch % print_step == 0 and self.is_rank0:
                logger.info(', '.join(epoch_info))
                print(', '.join(epoch_info)) # for multiple gpus

        total_train_time = time() - start

        # return early_stop.best_score, total_train_time
        # return self.es.best_score, total_train_time
        return best_result, total_train_time

    # def train_model_per_batch(self, batch_matrix, batch_weight=None):
    def train_model_per_batch(self, batch_matrix, item_input=None, e_start=False): ######
        # zero grad
        self.optimizer.zero_grad()

        # # model forwrad
        # output, kl_loss = self.forward(batch_matrix)
        #
        # # loss
        # # ce_loss = -(F.log_softmax(output, 1) * batch_matrix).mean()
        # if batch_weight is None:
        #     ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()
        # else:
        #     ce_loss = -((F.log_softmax(output, 1) * batch_matrix) * batch_weight.view(output.shape[0], -1)).sum(
        #         1).mean()
        #     # ce_loss = -((F.log_softmax(output, 1) * batch_matrix).sum(1) * batch_weight).mean()
        #
        # loss = ce_loss + kl_loss * self.anneal # kl loss?
        #
        # # backward
        # loss.backward()


        # check backward here
        # item_input = list(set(item_input))

        # users = torch.Tensor(np.arange(batch_matrix.shape[0])).int()
        # items = torch.Tensor(item_input).int()

        # users = np.arange(batch_matrix.shape[0])
        # items = item_input

        y_pred = self.forward(batch_matrix)
        # with torch.no_grad():
        #     y_prob = torch.softmax(y_pred, dim=1) # check softmax-cpr? dim 1, row sum = 1
        #     # y_arg = torch.argmax(y, dim=1)
        #
        #     # try sigmoid
        #     # y_prob = F.sigmoid(y_pred)
        #     loss = self.criterion(y_prob, batch_matrix)
        #     ba_loss = loss.item()
        # y_pred.backward((y_prob - batch_matrix) / batch_matrix.shape[0])  # for CrossEntropy Loss

        # y_prob = F.softmax(y_pred[:, items], dim=1)
        # y_prob = F.tanh(y_pred[:, items])
        # y_prob = F.sigmoid(y_pred[:, items])
        

        if len(y_pred.shape) == 1:
            # print("check out:", users, y_pred.shape, batch_matrix.shape)
            y_pred = y_pred.unsqueeze(0)
        

        # y_prob = F.softmax(y_pred, dim=1)
        # y_prob = y_prob[users, items]

        # y_prob = F.sigmoid(y_pred[users, items]) ######
        # y_prob = F.sigmoid(y_pred)

        # y_prob = F.sigmoid(y_pred[:, items]) ######
        # loss = self.criterion(y_prob, batch_matrix[:, items])

        # loss = -(batch_matrix[users, items] * torch.log(y_prob) + (1 - batch_matrix[users, items]) * torch.log(1 - y_prob)).sum()

        # loss = self.criterion(y_prob, batch_matrix[:, items])
        # loss = self.criterion(y_prob, batch_matrix)

        # loss = self.criterion(y_prob, batch_matrix[users, items])
        # loss = self.criterion(y_pred[users, items], batch_matrix[users, items])

        # y_pred = F.sigmoid(y_pred)
        loss = -(F.log_softmax(y_pred, 1) * batch_matrix).sum(1).mean() # ce loss (vae)
        # loss = -(F.log_softmax(y_pred, 1) * batch_matrix)[users, items].sum() ###
        # loss = -(F.log_softmax(y_pred, 1) * batch_matrix)[:, items].sum() ###

        # loss = -(F.log_softmax(y_pred * item_input, 1) * batch_matrix * item_input).sum(1).mean() # ce loss (vae)

        # if torch.isnan(loss).any():
        #     import pdb; pdb.set_trace()  # Enter the debugger
        #     has_nan = y_pred.isnan().any()
        #     print("y_pred nan val: ", has_nan)
        loss.backward()

        # step
        self.optimizer.step()

        for i, param in enumerate(self.parameters()):
            self.abs_gradient_max = max(self.abs_gradient_max, abs(torch.max(param.grad)))
            self.abs_gradient_avg = torch.sum(torch.abs(param.grad)) / (param.grad.numel())

        # ######
        # if e_start:
        #     self.clip_help()
        # else:
        #     self.clip()

        self.clip()

        # self.update_count += 1

        return loss
        # return loss.item()
        # return ba_loss


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