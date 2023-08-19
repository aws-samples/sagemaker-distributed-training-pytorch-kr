FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-gpu-py310
  
RUN pip install --no-cache-dir rouge_score \
    && pip install --no-cache-dir fire \
    && pip install --no-cache-dir sentencepiece \
    && pip install --no-cache-dir wandb \
    && pip install --no-cache-dir openai

RUN pip install --no-cache-dir transformers
RUN pip uninstall -y accelerate \
    && pip install --no-cache-dir accelerate>=0.20.3 
    
RUN pip uninstall -y deepspeed
# RUN pip install deepspeed

ARG DS_BUILD_OPS=0
ARG DS_BUILD_FUSED_ADAM=1

RUN mkdir -p /tmp && \
    cd /tmp && \
    git clone https://github.com/microsoft/DeepSpeed.git && \
    cd DeepSpeed && \
    pip install -r requirements/requirements-dev.txt && \
    pip install -r requirements/requirements.txt && \
    DS_BUILD_OPS=$DS_BUILD_OPS DS_BUILD_FUSED_ADAM=$DS_BUILD_FUSED_ADAM pip install -v . && \
    cd .. && \
    rm -rf DeepSpeed
 
WORKDIR /