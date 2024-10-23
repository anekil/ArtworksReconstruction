FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other commonly used data science packages
RUN pip3 install numpy pandas matplotlib scikit-learn comet_ml datasets optuna

# Set the working directory
WORKDIR /workspace

# Set the default command to bash
CMD ["/bin/bash"]