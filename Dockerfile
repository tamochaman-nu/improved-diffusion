# ベースイメージはbuild引数で切り替え可能。ホストのNVIDIAドライバが対応するCUDAバージョン
# (nvidia-smi右上の"CUDA Version")以下のものを選ぶこと。例:
#   ドライバ >= 550 (CUDA 12.4): pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel (デフォルト)
#   ドライバ >= 530 (CUDA 12.1): pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
#   ドライバ >= 520 (CUDA 11.8): pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel
# docker-compose.ymlからは.envのBASE_IMAGE / IMAGE_TAGで指定する(.env.example参照)。
ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

# Avoid interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - OpenMPI is required for mpi4py and distributed training
# - git is useful for cloning or interacting with repositories
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the setup files first for better caching
COPY setup.py /app/

# Install python dependencies
# Note: we include mpi4py and Pillow as they are used in the codebase
RUN pip install --no-cache-dir \
    blobfile>=1.0.5 \
    torch \
    tqdm \
    mpi4py \
    Pillow

# The rest of the codebase will be mounted via docker-compose,
# but we set the default command to bash for interactive usage.
CMD ["/bin/bash"]
