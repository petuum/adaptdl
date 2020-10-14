FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /root

COPY * /root/
RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r /root/requirements.txt