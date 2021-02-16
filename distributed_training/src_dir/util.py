
import codecs
import json
import logging
import os
import shutil
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models

from collections import OrderedDict

try:
    import dis_util
except ImportError:
    pass
# import sagemaker_containers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def torch_model(model_name,
                num_classes=0,
                pretrained=True,
                local_rank=0,
                model_parallel=False):
    #     model_names = sorted(name for name in models.__dict__
    #                          if name.islower() and not name.startswith("__")
    #                          and callable(models.__dict__[name]))

    if (model_name == "inception_v3"):
        raise RuntimeError(
            "Currently, inception_v3 is not supported by this example.")

    # create model
    if pretrained:
        print("=> using pre-trained model '{}'".format(model_name))
        if model_parallel:
            if local_rank == 0:
                model = models.__dict__[model_name](pretrained=True)
            dis_util.barrier()
        model = models.__dict__[model_name](pretrained=True)
    else:
        print("=> creating model '{}'".format(model_name))
        model = models.__dict__[model_name]()

    if num_classes > 0:
        n_inputs = model.fc.in_features

        # add more layers as required
        classifier = nn.Sequential(
            OrderedDict([('fc_output', nn.Linear(n_inputs, num_classes))]))

        model.fc = classifier

    return model


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_model(state, is_best, args):
    logger.info("Saving the model.")
    filename = os.path.join(args.model_dir, 'checkpoint.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(state, filename, _use_new_zipfile_serialization=False)

    if is_best:
        shutil.copyfile(filename, os.path.join(args.model_dir,
                                               'model_best.pth'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1**factor)
    
    # Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if args.rank == 0:
        print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_history(path, history):

    history_for_json = {}
    # transform float values that aren't json-serializable
    for key in history.keys():
        history_for_json[key] = list(map(float, history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history_for_json,
                  f,
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4)


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    elif hasattr(t, 'index'):
        return t[0]
    else:
        return t


def init_modelhistory(model_history):
    model_history['epoch'] = []
    model_history['batch_idx'] = []
    model_history['batch_time'] = []
    model_history['losses'] = []
    model_history['top1'] = []
    model_history['top5'] = []
    model_history['val_epoch'] = []
    model_history['val_batch_idx'] = []
    model_history['val_batch_time'] = []
    model_history['val_losses'] = []
    model_history['val_top1'] = []
    model_history['val_top5'] = []
    model_history['val_avg_epoch'] = []
    model_history['val_avg_batch_time'] = []
    model_history['val_avg_losses'] = []
    model_history['val_avg_top1'] = []
    model_history['val_avg_top5'] = []
    return model_history


def aws_s3_sync(source, destination):
    """aws s3 sync in quiet mode and time profile"""
    import time, subprocess
    cmd = ["aws", "s3", "sync", "--quiet", source, destination]
    print(f"Syncing files from {source} to {destination}")
    start_time = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    end_time = time.time()
    print("Time Taken to Sync: ", (end_time - start_time))
    return


def sync_local_checkpoints_to_s3(
    local_path="/opt/ml/checkpoints",
    s3_path=os.path.dirname(os.path.dirname(os.getenv('SM_MODULE_DIR', ''))) +
    '/checkpoints'):
    """ sample function to sync checkpoints from local path to s3 """

    import boto3, botocore
    #check if local path exists
    if not os.path.exists(local_path):
        raise RuntimeError(
            "Provided local path {local_path} does not exist. Please check")

    #check if s3 bucket exists
    s3 = boto3.resource('s3')
    if 's3://' not in s3_path:
        raise ValueError(
            "Provided s3 path {s3_path} is not valid. Please check")

    s3_bucket = s3_path.replace('s3://', '').split('/')[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise RuntimeError('S3 bucket does not exist. Please check')
    aws_s3_sync(local_path, s3_path)
    return


def sync_s3_checkpoints_to_local(
    local_path="/opt/ml/checkpoints",
    s3_path=os.path.dirname(os.path.dirname(os.getenv('SM_MODULE_DIR', ''))) +
    '/checkpoints'):
    """ sample function to sync checkpoints from s3 to local path """

    import boto3, botocore
    #creat if local path does not exists
    if not os.path.exists(local_path):
        print(f"Provided local path {local_path} does not exist. Creating...")
        try:
            os.makedirs(local_path)
        except Exception as e:
            raise RuntimeError(f"failed to create {local_path}")

    #check if s3 bucket exists
    s3 = boto3.resource('s3')
    if 's3://' not in s3_path:
        raise ValueError(
            "Provided s3 path {s3_path} is not valid. Please check")

    s3_bucket = s3_path.replace('s3://', '').split('/')[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise RuntimeError('S3 bucket does not exist. Please check')
    aws_s3_sync(s3_path, local_path)
    return
