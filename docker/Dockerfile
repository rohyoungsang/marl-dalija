#FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
MAINTAINER Tabish Rashid

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64
ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
#RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
#    rm -rf /var/lib/apt/lists/*

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install unzip
RUN pip3 install --upgrade pip

# Python packages we use (or used at one point...)
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger
RUN pip3 install pygame

# add rys
RUN pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install protobuf
RUN pip install absl-py==0.13.0
RUN pip install sacred numpy h5py scikit-learn PyYAML==5.4.1


RUN mkdir /install
WORKDIR /install

RUN pip3 install jsonpickle==0.9.6
# install Sacred (from OxWhirl fork)
RUN pip3 install setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
#RUN pip3 install torch
#RUN pip3 install torchvision snakeviz pytest probscale
RUN pip3 install snakeviz pytest probscale

# SMAC 설치
RUN pip3 install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH=/pymarl/3rdparty/StarCraftII

# StarCraft II 본체 설치
RUN mkdir -p /pymarl/3rdparty && \
    cd /pymarl/3rdparty && \
    if [ ! -d StarCraftII ]; then \
        wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip && \
        unzip -P iagreetotheeula SC2.4.10.zip && \
        rm SC2.4.10.zip; \
    else \
        echo "StarCraftII already exists. Skipping download."; \
    fi

# SMAC 맵 설치 (기존 맵 보존)
RUN mkdir -p /pymarl/3rdparty/StarCraftII/Maps && \
    cd /tmp && \
    wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip && \
    unzip SMAC_Maps.zip && \
    cp -rn SMAC_Maps/* /pymarl/3rdparty/StarCraftII/Maps/ && \
    rm -rf SMAC_Maps SMAC_Maps.zip

WORKDIR /pymarl
