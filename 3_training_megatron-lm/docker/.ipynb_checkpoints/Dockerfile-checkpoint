FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-gpu-py310
   
RUN git clone https://github.com/NVIDIA/apex \
  && cd apex \
  # recent commit, torch._six fix is included https://github.com/NVIDIA/apex/commit/a78ccf0b3e3f7130b3f157732dc8e8e651389922
  && git checkout 27de66c \
  && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
  && cd .. \
  
RUN pip install regex
 
WORKDIR /