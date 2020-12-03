FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /root

COPY examples/BERT/* /root/
COPY examples/requirements.txt /root/
COPY adaptdl adaptdl

ARG ADAPTDL_VERSION=0.0.0
RUN cd adaptdl && python3 setup.py bdist_wheel
RUN ADAPTDL_VERSION=${ADAPTDL_VERSION} pip install adaptdl/dist/*.whl
#RUN pip install torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install -r /root/requirements.txt
RUN python3 -m pip install torchtext==0.7.0
