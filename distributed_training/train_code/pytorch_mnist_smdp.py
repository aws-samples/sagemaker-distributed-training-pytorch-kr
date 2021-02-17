import argparse
import json
import logging
import os
# import sagemaker_containers
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms


########################################################
####### 1. SageMaker Distributed Data Parallel  ########
#######  - Import Package and Initialization    ########
########################################################

import smdistributed.dataparallel.torch.distributed as smdp
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as smdpDDP

if not smdp.is_initialized():
    smdp.init_process_group()
    
#######################################################


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _get_train_data_loader(args, is_distributed, **kwargs):
    logger.info("Get train data loader")
    dataset = datasets.MNIST(args.data_dir, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    
    #######################################################
    ####### 3. SageMaker Distributed Data Parallel  #######
    #######  - Add num_replicas and rank            #######
    #######################################################

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=args.rank) if is_distributed else None
    
    #######################################################
    
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler, **kwargs)


def _get_test_data_loader(args, **kwargs):
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


def train(args):
    is_distributed = args.world_size > 1
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device : {}".format(device))
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args, is_distributed, **kwargs)
    test_loader = _get_test_data_loader(args, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))
    
    
    model = Net().to(device)
    
    
    #######################################################
    ####### 4. SageMaker Distributed Data Parallel  #######
    #######  - Add num_replicas and rank            #######
    #######################################################    
    
    if is_distributed and use_cuda:
        torch.cuda.set_device(args.local_rank)
        model = smdpDDP(model)
        
    #######################################################    
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f},'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
        test(model, test_loader, device)
    save_model(model, args.model_dir)


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {:.0f}% ({}/{})\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset), correct, len(test_loader.dataset)))


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    # Setting for Distributed Training
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    
    # SageMaker Container environment
    parser.add_argument('--model-dir', type=str, default='../model')
    parser.add_argument('--data-dir', type=str, default='../data')
    
    args = parser.parse_args()
    
    try:
        args.model_dir = os.environ['SM_MODEL_DIR']
        args.data_dir = os.environ['SM_CHANNEL_TRAINING']
    except KeyError as e:
        print("The model starts training on the local host without SageMaker TrainingJob.")
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        pass

    ########################################################
    ####### 2. SageMaker Distributed Data Parallel   #######
    #######  - Get all number of GPU and rank number #######
    ########################################################
    
    args.world_size = smdp.get_world_size()  # all number of GPU
    args.rank = smdp.get_rank()              # total rank in all hosts
    args.local_rank = smdp.get_local_rank()  # rank per host
    
    ########################################################
    
    
    train(args)
