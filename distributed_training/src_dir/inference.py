
from __future__ import absolute_import

import argparse
import json
import logging
import os
import sys
import time
import random
from os.path import join
import numpy as np
import io
import tarfile

from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import copy
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms, models

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'


def model_fn(model_dir):
    """
    This function is called by the Pytorch container during hosting when running on SageMaker with
    values populated by the hosting environment.

    This function loads models written during training into `model_dir`.
    """
    
    print("****model_dir : {}".format(model_dir))
    traced_model = torch.jit.load(os.path.join(model_dir, 'model_result/model.pt'))
    return traced_model


def input_fn(request_body, request_content_type='application/x-image'):
    """This function is called on the byte stream sent by the client, and is used to deserialize the
    bytes into a Python object suitable for inference by predict_fn .
    """
    logger.info('An input_fn that loads a image tensor')
    if request_content_type == 'application/x-image':
        img = Image.open(io.BytesIO(request_body))
#         img_arr = np.array(Image.open(io.BytesIO(request_body)))
#         img = Image.fromarray(img_arr.astype('uint8')).convert('RGB')

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        img_data = transform(img)
        return img_data
    else:
        raise ValueError(
            'Requested unsupported ContentType in content_type : ' + request_content_type)


def predict_fn(input_data, model):
    """
    This function receives a NumPy array and makes a prediction on it using the model returned
    by `model_fn`.
    """
    logger.info('Entering the predict_fn function')
    input_data = input_data.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    result = {}
    with torch.no_grad():
        with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
            output = model(input_data)
            pred = output.max(1, keepdim=True)[1]
            result['output'] = output.numpy()
            result['pred'] = pred
    return result        


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    """This function is called on the return value of predict_fn, and is used to serialize the
    predictions back to the client.
    """
    return json.dumps({'result': prediction_output['output'].tolist(), 'pred': prediction_output['pred'].tolist()}), accept
