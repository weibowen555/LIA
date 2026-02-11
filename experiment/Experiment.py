import copy
from time import time
import torch
# from dataloader.DataBatcher import DataBatcher
import torch.distributed as dist
from evaluation import Evaluator
from scipy.sparse import csr_matrix
from utils import Config, Logger, ResultTable, make_log_dir
import torch.distributed as dist
from torch.utils.data import DataLoader

def train_model(model, dataset, evaluator, early_stop, logger, config):
    # train model with the given experimental setting
    logger.info('train start ... !')
    early_stop.initialize()
    test_score, train_time = model.train_model(dataset, evaluator, early_stop, logger, config)

    return test_score, train_time



def train_model_mp(gpu, args):
    # logger = args.logger
    dataset = args.dataset
    # evaluator = args.evaluator
    early_stop = args.early_stop
    config = args.config
    log_dir = args.log_dir

    global_rank = args.node_id * args.num_gpus + gpu

    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=global_rank)
    
    global_rank = torch.distributed.get_rank()
    device_id = args.device_ids[gpu]
    torch.cuda.set_device(device_id)

    # device_id = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")

    # For multi-node: only global rank 0 is the primary
    if global_rank == 0:
        is_rank0 = True
    else:
        is_rank0 = False


    dist.barrier()

    dataset_sample = copy.deepcopy(dataset)

    samples = torch.utils.data.distributed.DistributedSampler(dataset.train_matrix.toarray(), num_replicas=args.world_size, rank=global_rank)
    user_idx = list(samples)
    # print("check: ", device_id, user_idx)

    # train_loader = DataLoader(dataset.train_matrix.toarray(), batch_size=512, shuffle=False, pin_memory=True, sampler=samples)
    # for x in train_loader:
    #     print(torch.all(x == torch.FloatTensor(dataset.train_matrix.toarray()[user_idx])))

    # print("check new mat: ", dataset.train_matrix[user_idx])

    dataset_sample.train_matrix = dataset.train_matrix[user_idx]
    dataset_sample.vali_dict = {i: dataset_sample.vali_dict[u] for i, u in enumerate(user_idx)}
    dataset_sample.test_dict = {i: dataset_sample.test_dict[u] for i, u in enumerate(user_idx)}

    # dataset_sample.train_matrix = csr_matrix(torch.utils.data.distributed.DistributedSampler(dataset.train_matrix.toarray(), num_replicas=args.world_size, rank=global_rank))
    # dataset_sample.vali_dict = torch.utils.data.distributed.DistributedSampler(dataset.vali_dict, num_replicas=args.world_size, rank=global_rank)
    # dataset_sample.test_dict = torch.utils.data.distributed.DistributedSampler(dataset.test_dict, num_replicas=args.world_size, rank=global_rank)
    
    # dataset_sample.vali_dict = {index: val.copy() for index, val in enumerate(dataset_sample.vali_dict.values())}
    # dataset_sample.test_dict = {index: val.copy() for index, val in enumerate(dataset_sample.test_dict.values())}

    dataset_sample.num_users, dataset_sample.num_items = dataset_sample.train_matrix.shape[0], dataset_sample.train_matrix.shape[1]

    num_users, num_items = dataset_sample.num_users, dataset_sample.num_items

    ###
    test_eval_pos, test_eval_target, vali_eval_target, eval_neg_candidates = dataset_sample.test_data()
    # test_evaluator = Evaluator(test_eval_pos, test_eval_target, eval_neg_candidates, **config['Evaluator'], num_users=num_users, num_items=num_items, item_id = dataset.item_id_dict)
    test_evaluator = Evaluator(test_eval_pos, test_eval_target, vali_eval_target, eval_neg_candidates, **config['Evaluator'], num_users=num_users, num_items=num_items, item_id=None)
    evaluator = test_evaluator

    model_class = args.MODEL_CLASS
    model = model_class(dataset_sample, config['Model'], device_id, is_rank0, distributed=True)
    print("model initialized!!! ", device_id, is_rank0)

    # train model with the given experimental setting
    if global_rank == 0:
        logger = args.logger
        logger.info('train start ... !')
    else:
        logger = None
    print('train start ... !')
    early_stop.initialize()
    test_score, train_time = model.train_model(dataset_sample, evaluator, early_stop, logger, config)

    if global_rank == 0:
        print("output:", test_score)
        m, s = divmod(train_time, 60)
        h, m = divmod(m, 60)
        logger.info('\nTotal training time - %d:%d:%d(=%.1f sec)' % (h, m, s, train_time))

        # show result
        evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
        evaluation_table.add_row('Score', test_score)

        # evaluation_table.show()
        logger.info(evaluation_table.to_string())
        logger.info("Saved to %s" % (log_dir))

    # return test_score, train_time