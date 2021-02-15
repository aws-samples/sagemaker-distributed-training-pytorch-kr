
import argparse
import logging
import numpy as np
import os
import random
import sys
import shutil
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data.distributed
from torch.cuda.amp import autocast

import util

# smdist import package
try:
    # Import smdist PyTorch Modules
    import smdistributed.dataparallel.torch.distributed as sdp
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
    
    # SMP: Import SMP API
    import smdistributed.modelparallel.torch as smp

except ImportError:
    pass
#     raise ImportError("Please install smdist.")

try:
    from apex.parallel import DistributedDataParallel as apexDDP
    import torch.distributed as apex
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def dist_init(fn, args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

        if cudnn.deterministic:
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

    args.is_distributed = len(args.hosts) > 1 and args.backend is not None
    args.is_multigpus = args.num_gpus > 1
    args.multigpus_distributed = (args.is_distributed or args.is_multigpus)

    logger.debug("multigpus_distributed - {}".format(
        args.multigpus_distributed))
    logger.debug("Number of gpus available - {}".format(args.num_gpus))

    #     print("######### Start Training #########")

    if args.multigpus_distributed:
        if args.apex:
            # Initialize the distributed environment.
            mp.spawn(fn, nprocs=args.num_gpus, args=(args, ))
        else:
            if args.data_parallel and not sdp.is_initialized():
                sdp.init_process_group()
            elif args.model_parallel and not smp.is_initialized():
                smp.init()

            fn(None, args)

            if args.model_parallel:
                smp.barrier()
    else:
        fn(0, args)


#     return args


def dist_setting(args):
    #     args.data_parallel = False

    print("args.data_parallel : {}".format(args.data_parallel))
    print("args.model_parallel : {}".format(args.model_parallel))
    print("args.apex : {}".format(args.apex))

    args.world_size = 1
    args.host_num = args.hosts.index(args.current_host)

    if args.data_parallel:
        args.world_size = sdp.get_world_size()
        args.rank = sdp.get_rank()  # total rank in all hosts
        args.local_rank = sdp.get_local_rank()  # rank per host
    elif args.model_parallel:
        args.world_size = smp.size()
        args.local_rank = smp.local_rank()  # rank per host
        args.rank = smp.rank()
        args.dp_size = smp.dp_size()
        args.dp_rank = smp.dp_rank()
        print(
            "smp.rank() : {}, smp.size() : {}, smp.mp_rank() : {}, smp.local_size() : {}, smp.get_mp_group() : {}, smp.get_dp_group() : {}, smp.local_rank() : {}, smp.dp_size() : {}, smp.dp_rank() : {}"
            .format(smp.rank(), smp.size(), smp.mp_rank(), smp.local_size(),
                    smp.get_mp_group(), smp.get_dp_group(), smp.local_rank(),
                    smp.dp_size(), smp.dp_rank()))
    else:
        args.world_size = len(args.hosts) * args.num_gpus
        if args.local_rank is not None:
            args.rank = args.num_gpus * args.host_num + \
                args.local_rank  # total rank in all hosts

        dist.init_process_group(backend=args.backend,
                                rank=args.rank,
                                world_size=args.world_size)
        logger.info(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '
            .format(args.backend, dist.get_world_size()) +
            'Current host rank is {}. Number of gpus: {}'.format(
                dist.get_rank(), args.num_gpus))

    print("**** [dist_setting] args.rank : {}".format(args.rank))
    print("args.world_size : {}".format(args.world_size))
    print("Use GPU: {} for training".format(args.local_rank))

    args.lr = args.lr * float(args.world_size)

#     args.batch_size //= args.world_size // args.num_gpus
#     args.batch_size = max(args.batch_size, 1)

    return args


def dist_model(model, args):
    if args.multigpus_distributed:
        #     if args.sync_bn:
        # #         import apex
        #         print("using apex synced BN")
        #         model = apex.parallel.convert_syncbn_model(model)

        if args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)

            if not (args.apex or args.data_parallel or args.model_parallel):
                model.cuda(args.local_rank)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.rank])
        else:
            if not (args.apex or args.data_parallel or args.model_parallel):
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.rank is not None:
        torch.cuda.set_device(args.rank)
        if not (args.apex or args.data_parallel or args.model_parallel):
            model = model.cuda(args.rank)
    else:
        if not (args.apex or args.data_parallel or args.model_parallel):
            model = torch.nn.DataParallel(model).cuda()

    return model, args


def apex_init(model, optimizer, args):
    model = model.cuda()
    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level=args.opt_level,
        keep_batchnorm_fp32=args.keep_batchnorm_fp32,
        loss_scale=args.loss_scale)
    if args.multigpus_distributed:
        model = apexDDP(model, delay_allreduce=True)
    return model, optimizer, args


def sdp_init(model, optimizer, args):
    model = DDP(model.to(args.device), broadcast_buffers=False)
    #     model = DDP(model, device_ids=[args.rank], broadcast_buffers=False)
    model.cuda(args.local_rank)
    return model, optimizer, args


def smp_init(model, optimizer, args):
    model = smp.DistributedModel(model)
    args.scaler = smp.amp.GradScaler()
    optimizer = smp.DistributedOptimizer(optimizer)
    if args.partial_checkpoint:
        args.checkpoint = smp.load(args.partial_checkpoint, partial=True)
        model.load_state_dict(args.checkpoint["model_state_dict"])
        optimizer.load_state_dict(args.checkpoint["optimizer_state_dict"])
    elif args.full_checkpoint:
        args.checkpoint = smp.load(args.full_checkpoint, partial=False)
        model.load_state_dict(args.checkpoint["model_state_dict"])
        optimizer.load_state_dict(args.checkpoint["optimizer_state_dict"])

    return model, optimizer, args


def apex_loss(loss, optimizer):
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    #     print("rt : {}".format(rt))
    #     sdp.all_reduce(rt)
    #     print("args.world_size : {}".format(args.world_size))
    #     rt /= args.world_size
    return rt


def smp_lossgather(loss, args):
    if args.use_horovod or args.use_ddp:
        # Rubik: If using data parallelism, gather all losses across different model
        # replicas and check if losses match.

        losses = smp.allgather(loss, smp.DP_GROUP)
        for l in losses:
            assert math.isclose(l, losses[0])

        assert loss < 0.14
    else:
        assert loss < 0.08


def smp_savemodel(model, optimizer, is_best, args):
    filepath = '/opt/ml/local_checkpoints'
    filename = os.path.join(filepath, 'smp_full_checkpoint.pt')

    if args.rank == 0:
        if os.path.exists(filepath):
            print("-INFO- PATH DO EXIST")
        else:
            os.makedirs(filepath)
            print("-INFO- PATH DO NOT EXIST")
    smp.barrier()


    if args.dp_rank == 0:
        if args.save_full_model:
            model_dict = model.state_dict()
            opt_dict = optimizer.state_dict()
            smp.save(
                {
                    "model_state_dict": model_dict,
                    "optimizer_state_dict": opt_dict
                },
                filename,
                partial=False,
            )
        else:
            model_dict = model.local_state_dict()
            opt_dict = optimizer.local_state_dict()
            smp.save(
                {
                    "model_state_dict": model_dict,
                    "optimizer_state_dict": opt_dict
                },
                filename,
                partial=True,
            )
    smp.barrier()

    if args.rank == 0:
        print("Start syncing")
        base_s3_path = os.path.dirname(
            os.path.dirname(os.getenv('SM_MODULE_DIR', '')))
        curr_host = os.getenv('SM_CURRENT_HOST')
        full_s3_path = f'{base_s3_path}/checkpoints/{curr_host}/'
        util.sync_local_checkpoints_to_s3(local_path=filepath,
                                          s3_path=full_s3_path)
        print("Finished syncing")

        print("is_best : {}".format(is_best))
        if is_best:
            shutil.copyfile(filename,
                            os.path.join(args.model_dir, 'model_best.pth'))
    smp.barrier()


def barrier():
    smp.barrier()


try:
    # Rubik: Define smp.step. Return any tensors needed outside.
    @smp.step
    def train_step(model, criterion, input, target, scaler, args):
        with autocast(1 > 0):
            output = model(input)

        loss = criterion(output, target)

        loss = loss.mean()

        print("***** smp train_step : {}".format(loss))
        # scaled_loss = scaler.scale(loss) if args.amp else loss
        model.backward(loss)
        return output, loss

    # Rubik: Define smp.step for evaluation.
    @smp.step
    def test_step(model, criterion, input, target):
        output = model(input)
        loss = criterion(output, target)
        loss = loss.mean()
        print("***** smp test_step : {}".format(loss))
        return output, loss
except:
    pass
