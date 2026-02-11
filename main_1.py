# Import packages
import os
import sys
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

import utils.Constant as CONSTANT
from dataloader import UIRTDatset
from evaluation import Evaluator
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import warnings
import gc
import argparse

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_DEVICE_ORDER"] = "0, 1"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
# print(torch.cuda.device_count())

if __name__ == '__main__':
    mp.set_start_method('spawn')
    for _ in range(100):
        from experiment import EarlyStop, train_model, train_model_mp
        from utils import Config, Logger, ResultTable, make_log_dir#, set_random_seed

        # read configs
        config = Config(main_conf_path='./', model_conf_path='model_config')

        # apply system arguments if exist
        argv = sys.argv[1:]
        if len(argv) > 0:
            cmd_arg = OrderedDict()
            argvs = ' '.join(sys.argv[1:]).split(' ')
            for i in range(0, len(argvs), 2):
                arg_name, arg_value = argvs[i], argvs[i + 1]
                arg_name = arg_name.strip('-')
                cmd_arg[arg_name] = arg_value
            config.update_params(cmd_arg)

        ### old
        gpu = config.get_param('Experiment', 'gpu')
        gpu = str(gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        # os.environ["CUDA_DEVICE_ORDER"] = "0, 1"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        # device_ids = [0, 1]
        print("device: ", device)


        model_name = config.get_param('Experiment', 'model_name')

        # ### new
        # device_ids = config.get_param('Experiment', 'device_id')
        # num_gpus = len(device_ids)
        # num_nodes = config.get_param('Experiment', 'num_nodes')
        # WORLD_SIZE = num_gpus * num_nodes
        # node_id = config.get_param('Experiment', 'node_id')
        # os.environ['MASTER_ADDR'] = 'localhost' 
        # os.environ['MASTER_PORT'] = '8888'
        # # device = torch.cuda.set_device(device_id)

        # logger
        log_dir = make_log_dir(os.path.join('saves', model_name))
        logger = Logger(log_dir)
        config.save(log_dir)

        # dataset
        dataset_name = config.get_param('Dataset', 'dataset')
        dataset_type = CONSTANT.DATASET_TO_TYPE[dataset_name]
        dataset = UIRTDatset(**config['Dataset'])

        # evaluator - old
        num_users, num_items = dataset.num_users, dataset.num_items

        ###
        test_eval_pos, test_eval_target, vali_eval_target, eval_neg_candidates = dataset.test_data()
        # test_evaluator = Evaluator(test_eval_pos, test_eval_target, eval_neg_candidates, **config['Evaluator'], num_users=num_users, num_items=num_items, item_id = dataset.item_id_dict)
        test_evaluator = Evaluator(test_eval_pos, test_eval_target, vali_eval_target, eval_neg_candidates, **config['Evaluator'], num_users=num_users, num_items=num_items, item_id=None)
        
        # early stop
        early_stop = EarlyStop(**config['EarlyStop'])

        # Save log & dataset config.
        logger.info(config)
        logger.info(dataset)


        import model
        MODEL_CLASS = getattr(model, model_name)

        # # seed = config.get_param('Experiment', 'seed')

        # # build model
        # # set_random_seed(seed)


        ### old
        model = MODEL_CLASS(dataset, config['Model'], device)
        # train
        test_score, train_time = train_model(model, dataset, test_evaluator, early_stop, logger, config) ### old

        
        # ### new
        # print("parse args")
        # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # args = parser.parse_args()
        # args.MODEL_CLASS = MODEL_CLASS
        # args.dataset = dataset
        # args.early_stop = early_stop
        # args.log_dir = log_dir
        # args.logger = logger
        # args.config = config
        # args.node_id = node_id
        # args.world_size = WORLD_SIZE
        # args.num_gpus = num_gpus
        # args.device_ids = device_ids
        # # args = (MODEL_CLASS, dataset, early_stop, logger, config, node_id, WORLD_SIZE, num_gpus, device_id)
        # mp.spawn(train_model_mp, nprocs=num_gpus, args=(args,))


        ### old output info
        m, s = divmod(train_time, 60)
        h, m = divmod(m, 60)
        logger.info('\nTotal training time - %d:%d:%d(=%.1f sec)' % (h, m, s, train_time))

        # show result
        evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
        evaluation_table.add_row('Score', test_score)

        # evaluation_table.show()
        logger.info(evaluation_table.to_string())
        logger.info("Saved to %s" % (log_dir))

        # Extract global model
        if 'LOCA' not in model_name and (model_name == 'MultVAE' or model_name == 'EASE'):
            output = model.get_output(dataset)

            output_dir = os.path.join(dataset.data_dir, dataset.data_name, 'output')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_file = os.path.join(output_dir, model_name + '_output.p')
            with open(output_file, 'wb') as f:
                pickle.dump(output, f, protocol=4)
            config.save(output_dir)
            print(f"{model_name} output extracted!")

        # Extract Embedding
        if model_name == 'MultVAE':
            user_embedding = model.user_embedding(test_eval_pos)

            emb_dir = os.path.join(dataset.data_dir, dataset.data_name, 'embedding')
            if not os.path.exists(emb_dir):
                os.mkdir(emb_dir)
            emb_file = os.path.join(emb_dir, model_name + '_user.p')
            with open(emb_file, 'wb') as f:
                pickle.dump(user_embedding, f, protocol=4)
            config.save(emb_dir)
            print(f"{model_name} embedding extracted!")

        del model
        gc.collect()
        torch.cuda.empty_cache()