FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    git \
    sudo 

# Install Miniforge as root in /root's home directory
ENV CONDA_DIR=/root/miniforge3
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p ${CONDA_DIR} && \
    rm Miniforge3-$(uname)-$(uname -m).sh

# Fixes opencv issue
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set environment path for root
ENV PATH="${CONDA_DIR}/bin:$PATH"

# Initialize conda for bash and disable auto activation of base
RUN conda init bash && \
    echo "conda config --set auto_activate_base false" >> ~/.bashrc
WORKDIR /depth_any_camera
RUN git clone https://github.com/yuliangguo/depth_any_camera /depth_any_camera && \
cd /depth_any_camera && \
conda create -n dac python=3.9 -y 

SHELL ["conda", "run", "-n", "dac", "/bin/bash", "-c"]
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN cd /depth_any_camera && pip install -r requirements.txt 
