
import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
import cv2
from typing import Callable, cast

from albumentations import (
    RandomResizedCrop, HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE,
    RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion,
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise,
    MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, Resize, VerticalFlip,
    HorizontalFlip, CenterCrop, Normalize)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import SplitDataset
import webdataset as wds

import dis_util
import util

# print("######### Start Training #########")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class AlbumentationImageDataset(Dataset):
    def __init__(self, image_path, transform, args, check_img=None):
        self.image_path = image_path
        self.transform = transform
        self.args = args
        self.check_img = check_img
        self.image_list = self._loader_file(self.image_path, self.check_img)

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):

        image = cv2.imread(self.image_list[i][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Augment an image
        transformed = self.transform(image=image)["image"]
        transformed_image = np.transpose(transformed,
                                         (2, 0, 1)).astype(np.float32)
        return torch.tensor(transformed_image,
                            dtype=torch.float), self.image_list[i][1]

    def _loader_file(self, image_path, check_img):
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                      '.tiff', '.webp')

        def is_valid_file(x: str) -> bool:
            return x.lower().endswith(extensions)

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        self.classes = [d.name for d in os.scandir(image_path) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {
            cls_name: i
            for i, cls_name in enumerate(self.classes)
        }

        instances = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(image_path, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir,
                                                  followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)

                    if is_valid_file(path):
                        not_insert = False
                        if check_img:
                            try:
                                image = cv2.imread(path)
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                not_insert = False
                            except:
                                not_insert = True
                                pass
                        if not not_insert:
                            item = path, class_index
                            instances.append(item)
        return instances


def args_fn():
    parser = argparse.ArgumentParser(description='PyTorch Resnet50 Example')

    # Default Setting
    parser.add_argument(
        '--log-interval',
        type=int,
        default=5,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help=
        'backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)'
    )
    parser.add_argument('--channels-last', type=bool, default=True)
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-p',
                        '--print-freq',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')

    # Hyperparameter Setting
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=200,
                        metavar='N',
                        help='input batch size for testing (default: 200)')

    # Setting for Distributed Training
    parser.add_argument('--data_parallel', type=bool, default=False)
    parser.add_argument('--model_parallel', type=bool, default=False)
    parser.add_argument('--apex', type=bool, default=False)
    parser.add_argument('--opt-level', type=str, default='O0')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--sync_bn',
                        action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--prof',
                        default=-1,
                        type=int,
                        help='Only run 10 iterations for profiling.')

    # Setting for Model Parallel
    parser.add_argument("--horovod", type=int, default=0)
    parser.add_argument('--mp_parameters', type=str, default='')
    parser.add_argument("--ddp", type=int, default=0)
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--save_full_model", type=bool, default=True)
    parser.add_argument("--pipeline", type=str, default="interleaved")
    parser.add_argument("--assert-losses", type=int, default=0)
    parser.add_argument("--partial-checkpoint",
                        type=str,
                        default="",
                        help="The checkpoint path to load")
    parser.add_argument("--full-checkpoint",
                        type=str,
                        default="",
                        help="The checkpoint path to load")
    parser.add_argument("--save-full-model",
                        action="store_true",
                        default=False,
                        help="For Saving the current Model")
    parser.add_argument(
        "--save-partial-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    # SageMaker Container environment
    parser.add_argument('--hosts',
                        type=list,
                        default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host',
                        type=str,
                        default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir',
                        type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus',
                        type=int,
                        default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--output_data_dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--rank', type=int, default=0)

    args = parser.parse_args()
    return args


def _get_train_data_loader(args, **kwargs):

    transform = Compose([
        RandomResizedCrop(args.height, args.width),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        VerticalFlip(p=0.5),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ],
              p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ],
              p=0.3),
        HueSaturationValue(p=0.3),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ],
                        p=1.0)

    train_sampler = None
    train_dataloader = None

    dataset = AlbumentationImageDataset(image_path=os.path.join(
        args.data_dir, 'train'),
                                        transform=transform,
                                        args=args,
                                        check_img=True)

    drop_last = args.model_parallel

    train_sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=int(args.world_size), rank=int(
            args.rank)) if args.multigpus_distributed else None
    train_dataloader = data.DataLoader(dataset,
                                       batch_size=args.batch_size,
                                       shuffle=train_sampler is None,
                                       sampler=train_sampler,
                                       drop_last=drop_last,
                                       **kwargs)
    return train_dataloader, train_sampler


def _get_test_data_loader(args, **kwargs):
    logger.info("Get test data loader")
    transform = Compose([
        Resize(args.height, args.width),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    image_path = os.path.join(args.data_dir, 'val')
    dataset = AlbumentationImageDataset(image_path=image_path,
                                        transform=transform,
                                        args=args,
                                        check_img=True)

    drop_last = args.model_parallel
    print("test drop_last : {}".format(drop_last))
    test_sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=int(args.world_size), rank=int(
            args.rank)) if args.multigpus_distributed else None

    return data.DataLoader(dataset,
                           batch_size=args.test_batch_size,
                           shuffle=False,
                           sampler=test_sampler,
                           drop_last=drop_last)


def train(local_rank, args):
    best_acc1 = -1
    model_history = {}
    model_history = util.init_modelhistory(model_history)
    train_start = time.time()

    if local_rank is not None:
        args.local_rank = local_rank

    # distributed_setting
    if args.multigpus_distributed:
        args = dis_util.dist_setting(args)

    # choose model from pytorch model_zoo
    model = util.torch_model(
        args.model_name,
        num_classes=args.num_classes,
        pretrained=True,
        local_rank=args.local_rank,
        model_parallel=args.model_parallel)  # 1000 resnext101_32x8d
    criterion = nn.CrossEntropyLoss().cuda()

    model, args = dis_util.dist_model(model, args)

    # CuDNN library will benchmark several algorithms and pick that which it found to be fastest
    cudnn.benchmark = False if args.seed else True

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.apex:
        model, optimizer, args = dis_util.apex_init(model, optimizer, args)
    elif args.model_parallel:
        model, optimizer, args = dis_util.smp_init(model, optimizer, args)
    elif args.data_parallel:
        model, optimizer, args = dis_util.sdp_init(model, optimizer, args)

    train_loader, train_sampler = _get_train_data_loader(args, **args.kwargs)

    logger.info("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)))

    test_loader = _get_test_data_loader(args, **args.kwargs)

    #     if args.rank == 0:
    logger.info("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)))

    print(" local_rank : {}, local_batch_size : {}".format(
        local_rank, args.batch_size))

    for epoch in range(1, args.num_epochs + 1):
        ##
        batch_time = util.AverageMeter('Time', ':6.3f')
        data_time = util.AverageMeter('Data', ':6.3f')
        losses = util.AverageMeter('Loss', ':.4e')
        top1 = util.AverageMeter('Acc@1', ':6.2f')
        top5 = util.AverageMeter('Acc@5', ':6.2f')
        progress = util.ProgressMeter(
            len(train_loader), [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        model.train()
        end = time.time()

        # Set epoch count for DistributedSampler
        if args.multigpus_distributed and not args.model_parallel:
            train_sampler.set_epoch(epoch)

        for batch_idx, (input, target) in enumerate(train_loader):
            input = input.to(args.device)
            target = target.to(args.device)
            batch_idx += 1

            if args.model_parallel:
                output, loss = dis_util.train_step(model, criterion, input,
                                                   target, args.scaler, args)
                # Rubik: Average the loss across microbatches.
                loss = loss.reduce_mean()

            else:
                output = model(input)
                loss = criterion(output, target)

            if not args.model_parallel:
                # compute gradient and do SGD step
                optimizer.zero_grad()

            if args.apex:
                dis_util.apex_loss(loss, optimizer)
            elif not args.model_parallel:
                loss.backward()

            optimizer.step()

            if args.model_parallel:
                # compute gradient and do SGD step
                optimizer.zero_grad()

            if args.rank == 0:
                #             if args.rank == 0 and batch_idx % args.log_interval == 1:
                # Every print_freq iterations, check the loss, accuracy, and speed.
                # For best performance, it doesn't make sense to print these metrics every
                # iteration, since they incur an allreduce and some host<->device syncs.

                if args.model_parallel:
                    output = torch.cat(output.outputs)

                # Measure accuracy
                prec1, prec5 = util.accuracy(output, target, topk=(1, 5))

                # to_python_float incurs a host<->device sync
                losses.update(util.to_python_float(loss), input.size(0))
                top1.update(util.to_python_float(prec1), input.size(0))
                top5.update(util.to_python_float(prec5), input.size(0))

                # Waiting until finishing operations on GPU (Pytorch default: async)
                torch.cuda.synchronize()
                batch_time.update((time.time() - end) / args.log_interval)
                end = time.time()

                #                 if args.rank == 0:
                print(
                    'Epoch: [{0}][{1}/{2}] '
                    'Train_Time={batch_time.val:.3f}: avg-{batch_time.avg:.3f}, '
                    'Train_Speed={3:.3f} ({4:.3f}), '
                    'Train_Loss={loss.val:.10f}:({loss.avg:.4f}), '
                    'Train_Prec@1={top1.val:.3f}:({top1.avg:.3f}), '
                    'Train_Prec@5={top5.val:.3f}:({top5.avg:.3f})'.format(
                        epoch,
                        batch_idx,
                        len(train_loader),
                        args.world_size * args.batch_size / batch_time.val,
                        args.world_size * args.batch_size / batch_time.avg,
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5))

        acc1 = validate(test_loader, model, criterion, epoch, model_history,
                        args)

        is_best = False

        if args.rank == 0:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

        if not args.multigpus_distributed or (args.multigpus_distributed
                                              and not args.model_parallel
                                              and args.rank == 0):
            model_history['epoch'].append(epoch)
            model_history['batch_idx'].append(batch_idx)
            model_history['batch_time'].append(batch_time.val)
            model_history['losses'].append(losses.val)
            model_history['top1'].append(top1.val)
            model_history['top5'].append(top5.val)

            util.save_history(
                os.path.join(args.output_data_dir, 'model_history.p'),
                model_history)
            util.save_model(
                {
                    'epoch': epoch + 1,
                    'model_name': args.model_name,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'class_to_idx': train_loader.dataset.class_to_idx,
                }, is_best, args)
        elif args.model_parallel:
            if args.rank == 0:
                util.save_history(
                    os.path.join(args.output_data_dir, 'model_history.p'),
                    model_history)
            dis_util.smp_savemodel(model, optimizer, is_best, args)


def validate(val_loader, model, criterion, epoch, model_history, args):
    batch_time = util.AverageMeter('Time', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4e')
    top1 = util.AverageMeter('Acc@1', ':6.2f')
    top5 = util.AverageMeter('Acc@5', ':6.2f')
    progress = util.ProgressMeter(len(val_loader),
                                  [batch_time, losses, top1, top5],
                                  prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    end = time.time()

    #     print("**** validate *****")
    test_losses = []
    for batch_idx, (input, target) in enumerate((val_loader)):
        input = input.to(args.device)
        target = target.to(args.device)

        batch_idx += 1
        # compute output
        with torch.no_grad():
            if args.model_parallel:
                output, loss = dis_util.test_step(model, criterion, input,
                                                  target)
                loss = loss.reduce_mean()
                test_losses.append(loss)
            else:
                output = model(input)
                loss = criterion(output, target)

        # measure accuracy and record loss
        if args.model_parallel:
            output = torch.cat(output.outputs)

        prec1, prec5 = util.accuracy(output, target, topk=(1, 5))

        losses.update(util.to_python_float(loss), input.size(0))
        top1.update(util.to_python_float(prec1), input.size(0))
        top5.update(util.to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #         print("Validation args.rank : {}".format(args.rank))
        # TODO:  Change timings to mirror train().
        if args.rank == 0:
            print('Test: [{0}/{1}]  '
                  'Test_Time={batch_time.val:.3f}:({batch_time.avg:.3f}), '
                  'Test_Speed={2:.3f}:({3:.3f}), '
                  'Test_Loss={loss.val:.4f}:({loss.avg:.4f}), '
                  'Test_Prec@1={top1.val:.3f}:({top1.avg:.3f}), '
                  'Test_Prec@5={top5.val:.3f}:({top5.avg:.3f})'.format(
                      batch_idx,
                      len(val_loader),
                      args.world_size * args.batch_size / batch_time.val,
                      args.world_size * args.batch_size / batch_time.avg,
                      batch_time=batch_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))
            model_history['val_epoch'].append(epoch)
            model_history['val_batch_idx'].append(batch_idx)
            model_history['val_batch_time'].append(batch_time.val)
            model_history['val_losses'].append(losses.val)
            model_history['val_top1'].append(top1.val)
            model_history['val_top5'].append(top5.val)

    model_history['val_avg_epoch'].append(epoch)
    model_history['val_avg_batch_time'].append(batch_time.avg)
    model_history['val_avg_losses'].append(losses.avg)
    model_history['val_avg_top1'].append(top1.avg)
    model_history['val_avg_top5'].append(top5.avg)

    if args.assert_losses:
        dist_util.smp_lossgather(losses.avg, args)
    return top1.avg


def main():
    print("start main function")
    args = args_fn()
    print(
        "args.data_parallel : {} , args.model_parallel : {}, args.apex : {} , args.num_gpus : {}, args.num_classes"
        .format(args.data_parallel, args.model_parallel, args.apex,
                args.num_gpus, args.num_classes))

    args.use_cuda = int(args.num_gpus) > 0

    args.kwargs = {
        'num_workers': 16,
        'pin_memory': True
    } if args.use_cuda else {}
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    args = dis_util.dist_init(train, args)


if __name__ == '__main__':
    main()
