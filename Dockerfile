FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

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
