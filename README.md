# Amazon SageMaker Distributed Training

This repo introduces training scripts that run distributed training on an Amazon SageMaker. 

- [1_training_mnist_ddp.ipynb](1_training_mnist_ddp.ipynb) : Pytorch/SageMaker Distributed Data Parallel with MNIST dataset
- [2_training_oxford-pet_ddp.ipynb](2_training_oxford-pet_ddp.ipynb) : Pytorch/SageMaker Distributed Data Parallel with OXFORD-PET dataset
- [3_training_megatron-lm.ipynb](3_training_megatron-lm.ipynb) : [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) with [CodeParrot](https://github.com/huggingface/blog/blob/main/megatron-training.md#data-preprocessing) 
- [4_training_alpaca_deepspeed.ipynb](4_training_alpaca_deepspeed.ipynb) : [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) using [DeepSpeed](https://www.deepspeed.ai/tutorials/advanced-install/) with [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

## SageMaker Training Overview
With the increasing demand for machine learning models, the need for efficient and scalable training methods has become crucial. Distributed training allows users to train their models on multiple machines simultaneously, reducing the training time significantly.

Amazon SageMaker, a fully managed machine learning service, offers seamless integration with distributed training capabilities. It provides users with easy-to-implement ultra-clusters that leverage the power of distributed computing, enabling them to accelerate their model training process.

In conclusion, the distributed training for Amazon SageMaker offers a comprehensive and efficient solution for training deep learning models. By harnessing the power of distributed computing, users can significantly reduce training time and improve the accuracy of their models, ultimately leading to enhanced productivity and success in the field of machine learning.

### ADD IAM Role

Go to the [IAM Role Console](https://console.aws.amazon.com/iam/#/roles), search for and find the IAM Role used in the lab. if you created an IAM Role with the same name as ***SageMakerIamRole*** during the lab, please add ***AmazonEC2ContainerRegistryFullAccess*** to the ***SageMakerIamRole*** for pushing custom training docker image.


## Contributors
- Youngjoon Choi (choijoon@amazon.com)
- Daekeun Kim (daekeun@amazon.com)