# Amazon SageMaker Distributed Training (Image Classification for Oxford-IIIT Pet Dataset)

### Training/Deploying Model for Image dataset

### 1. 실습 구성
이번 실습에서는 아래 단계를 걸쳐서 진행을 할 예정입니다.
![fig1.png](figs/images_4/fig1.png)

### 2. 데이터셋 설명
Oxford-IIIT Pet Dataset은 37개 다른 종의 개와 고양이 이미지를 각각 200장 씩 제공하고 있으며, Ground Truth 또한 Classification, Object Detection, Segmentation와 관련된 모든 정보가 있으나, 이번 학습에서는 37개 class에 대해 일부 이미지로 Classification 문제를 해결하기 위해 학습을 진행할 예정입니다.

![fig3.jpg](figs/images_4/fig3.jpg)


### 3. 실습 수행 과정
![fig2.png](figs/images_4/fig2.png)

이번 실습은 SageMaker의 training job을 여러 개 띄워서 분산 학습이 가능하도록 구성하였습니다. 또한, GPU를 여러 개 가지고 있는 ml.p3.8xlarge, ml.p3.16xlarge, ml.p3dn.24xlarge, ml.p4dn.24xlarge를 함께 사용할 때에는 모든 GPU가 Training에서 활용될 수 있도록 구성하였습니다. [SageMaker Distributed training](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/distributed-training.html)은 [Data Parallel](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/data-parallel-intro.html)과 [Model Parallel](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/model-parallel.html) 2가지 방법을 지원하며, 기존 Distributed Training 보다 AWS의 인프라에 적합하게 구성하였기에 성능 또한 우수합니다. [Horovod](https://distributed-training-workshop.go-aws.com/)와 [APEX](https://github.com/NVIDIA/apex) (A Pytorch EXtension) 패키지와 같은 기존의 Distributed training도 수행이 가능합니다. 이번 실습에서는 SageMaker Data Parallel과 APEX 패키지를 모두 실행할 수 있도록 distributed training 환경을 구성하였으며, 실습을 통해 2개의 성능과 속도 등을 비교해 보도록 하겠습니다.

Training이 완료된 이후에는 학습된 model을 SageMaker Endpoint를 이용하여 deploy를 할 예정입니다. 이 때 GPU 대신 가격이 저렴한 CPU로 deploy를 하게 되면 Amazon Elastic Inference를 이용하여 inference 속도를 CPU보다는 더욱 빠르게 수행할 수 있도록 합니다.


## 실습 종료 후 리소스 정리

실습이 종료되면, 실습에 사용된 리소스들을 모두 삭제해 주셔야 불필요한 과금을 피하실 수 있습니다.

아래 삭제에 앞서 SageMaker Notebook을 통해 생성한 ***SageMaker Endpoint***를 각 Notebook 생성 페이지에서 SDK 명령어를 통해 삭제해 주시기 바랍니다.

### IAM Role 삭제

[IAM의 Role 콘솔](https://console.aws.amazon.com/iam/#/roles)로 이동하고 실습에 사용했던 IAM Role을 검색하여 찾은 후, ***delete***를 클릭하여 삭제합니다. 예를 들어 ***SageMakerIamRole***과 같은 이름으로 실습 과정에서 IAM Role을 생성하셨다면 이것을 찾아서 삭제합니다.

### SageMaker Notebook 삭제

[SageMaker 콘솔](https://ap-northeast-2.console.aws.amazon.com/sagemaker/home?region=ap-northeast-2#/dashboard)로 이동하고 실습에 사용했던 Notebook instance를 검색하여 찾은 후, ***delete***를 클릭하여 삭제합니다. 예를 들어 ***sagemaker-hol-lab***과 같은 이름으로 실습 과정에서 Notebook을 생성하셨다면 이것을 찾아서 삭제합니다.

### S3 Bucket 삭제

[S3 콘솔](https://s3.console.aws.amazon.com/s3/home?region=ap-northeast-2)로 이동하고 실습에 사용했던 2개의 bucket을 검색하여 찾은 후, ***delete***를 클릭하여 삭제합니다. 예를 들어 ***sagemaker-experiments-ap-northeast-2***와 ***sagemaker-ap-northeast-2*** 같은 이름으로 실습 과정에서 S3 Bucket을 생성하셨다면 이것을 찾아서 삭제합니다.

수고하셨습니다.\
이제 모든 리소스 삭제를 완료하셨습니다.


## Contributors
- Youngjoon Choi (choijoon@amazon.com)
- Daekeun Kim (daekeun@amazon.com)