FROM rayproject/ray-ml:1.7.0-py38-gpu
RUN sudo apt update && sudo apt install -y awscli
RUN python3 -m pip install adaptdl adaptdl_sched tensorboard boto3 aiohttp requests
RUN wget https://aws-s3-plugin.s3.us-west-2.amazonaws.com/binaries/0.0.1/1c3e69e/awsio-0.0.1-cp38-cp38-manylinux1_x86_64.whl && python3 -m pip install awsio-0.0.1-cp38-cp38-manylinux1_x86_64.whl && rm -f awsio-0.0.1-cp38-cp38-manylinux1_x86_64.whl
RUN python3 -m pip install pymoo==0.4.0
