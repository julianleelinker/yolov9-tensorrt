# syntax = docker/dockerfile:1.2
FROM nvcr.io/nvidia/tensorrt:22.12-py3

RUN pip install --upgrade pip
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install onnxruntime==1.16.3
RUN pip install onnx==1.15.0
RUN pip install requests
RUN pip install nvidia-pyindex
RUN pip install onnx-graphsurgeon
RUN pip install opencv-python-headless
RUN pip install pandas
RUN pip install IPython
RUN pip install psutil
RUN pip install pyyaml
RUN pip install tqdm
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install ipdb
