import argparse
import logging
import os
import sys

import time
import cv2

from typing import Callable, cast

from albumentations import (
    RandomResizedCrop, CLAHE, Transpose, ShiftScaleRotate, Blur, OpticalDistortion,
    HueSaturationValue, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast,
    Sharpen, Emboss, Flip, OneOf, Compose, Resize, VerticalFlip, Normalize)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torch.optim as optim
import torch.utils.data as data

import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

# import dis_util
import util

# print("######### Start Training #########")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class AlbumentationImageDataset(data.Dataset):
    def __init__(self, image_path, transform, args):
        self.image_path = image_path
        self.transform = transform
        self.args = args
        self.image_list = self._loader_file(self.image_path)

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

    def _loader_file(self, image_path):
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

    args = parser.parse_args()
    return args


def _get_train_data_loader(args):

    transform = Compose([
        RandomResizedCrop(args.height, args.width),
        GaussNoise(p=0.2),
        VerticalFlip(p=0.5),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ],p=1.0)

    train_sampler = None
    train_dataloader = None

    dataset = AlbumentationImageDataset(
        image_path=os.path.join(args.data_dir, 'train'),
        transform=transform,
        args=args
    )
    
    train_sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=args.world_size, rank=args.rank)
    
    train_dataloader = data.DataLoader(dataset,
                                       batch_size=args.batch_size,
                                       shuffle=train_sampler is None,
                                       sampler=train_sampler,
                                       num_workers=2,
                                       pin_memory=True)
    
    return train_dataloader, train_sampler


def _get_test_data_loader(args):
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
                                        args=args)

    return data.DataLoader(dataset,
                           batch_size=args.test_batch_size,
                           shuffle=False)


def dist_setting(args):
    
    if args.backend == 'smddp':
        import smdistributed.dataparallel.torch.torch_smddp
    
    backend = "gloo" if not torch.cuda.is_available() else args.backend
    
    args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    
    dist.init_process_group(backend=backend,
                            rank=args.rank,
                            world_size=args.world_size)

    args.batch_size //= args.world_size // 8
    args.batch_size = max(args.batch_size, 1)
    return args


def check_sagemaker(args):
    ## SageMaker
    if os.environ.get('SM_MODEL_DIR') is not None:
        args.data_dir = os.environ['SM_CHANNEL_TRAINING']
        args.model_dir = os.environ['SM_MODEL_DIR']
    return args
    

def train(args):
    best_acc1 = -1
    model_history = {}
    model_history = util.init_modelhistory(model_history)
    train_start = time.time()

    # choose model from pytorch model_zoo
    model = util.torch_model(
        args.model_name,
        num_classes=args.num_classes,
        pretrained=True,
        # pretrained=False
    )  # 1000 resnext101_32x8d
    
    criterion = nn.CrossEntropyLoss().cuda()

    torch.cuda.set_device(args.local_rank)
    model = DDP(model.to(args.device),
                device_ids=[args.local_rank],
                output_device=args.local_rank)
    
    # CuDNN library will benchmark several algorithms and pick that which it found to be fastest
    cudnn.benchmark = False if args.seed else True

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_loader, train_sampler = _get_train_data_loader(args)

    logger.info("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)))

    test_loader = _get_test_data_loader(args)

    #     if args.rank == 0:
    logger.info("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)))

    print(" local_rank : {}, local_batch_size : {}".format(
        args.local_rank, args.batch_size))

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

        train_sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(args.device)
            target = target.to(args.device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            if args.rank == 0:
                #             if args.rank == 0 and batch_idx % args.log_interval == 1:
                # Every print_freq iterations, check the loss, accuracy, and speed.
                # For best performance, it doesn't make sense to print these metrics every
                # iteration, since they incur an allreduce and some host<->device syncs.

                # Measure accuracy
                prec1, prec5 = util.accuracy(output, target, topk=(1, 5))

                # to_python_float incurs a host<->device sync
                losses.update(util.to_python_float(loss), data.size(0))
                top1.update(util.to_python_float(prec1), data.size(0))
                top5.update(util.to_python_float(prec5), data.size(0))

                # Waiting until finishing operations on GPU (Pytorch default: async)
                torch.cuda.synchronize()
                batch_time.update((time.time() - end) / args.log_interval)
                end = time.time()

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

                if args.rank == 0:
                    model_history['epoch'].append(epoch)
                    model_history['batch_idx'].append(batch_idx)
                    model_history['batch_time'].append(batch_time.val)
                    model_history['losses'].append(losses.val)
                    model_history['top1'].append(top1.val)
                    model_history['top5'].append(top5.val)


        acc1 = validate(test_loader, model, criterion, epoch, model_history,
                        args)

        is_best = False

        if args.rank == 0:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            util.save_history(
                os.path.join(args.model_dir, 'model_history.p'),
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

        dist.barrier() 


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
    for batch_idx, (data, target) in enumerate((val_loader)):
        data = data.to(args.device)
        target = target.to(args.device)

        # compute output
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        prec1, prec5 = util.accuracy(output, target, topk=(1, 5))

        losses.update(util.to_python_float(loss), data.size(0))
        top1.update(util.to_python_float(prec1), data.size(0))
        top5.update(util.to_python_float(prec5), data.size(0))

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

    return top1.avg

def main():
    print("start main function")
    args = args_fn()
    args = check_sagemaker(args)
    args = dist_setting(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train(args)

if __name__ == '__main__':
    main()
