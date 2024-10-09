# Use Nvidia CUDA base image
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    make \
    gcc \
    g++ \
    parallel \
    libncurses5-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev && \
    apt-get clean

# Create directories for tools and work
RUN mkdir -p /tools /work

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -O /tools/miniconda3.sh && \
    bash /tools/miniconda3.sh -b -p /tools/miniconda3 && \
    rm /tools/miniconda3.sh

# Add Miniconda to the PATH
ENV PATH="/tools/miniconda3/bin:${PATH}"

# Initialize Conda and create the PyTorch environment
RUN /tools/miniconda3/bin/conda init bash && \
    /tools/miniconda3/bin/conda create -n pytorch python=3.9 -y

# Install longcallR-nn
RUN git clone https://github.com/huangnengCSU/longcallR-nn.git /tools/longcallR-nn && \
    cd /tools/longcallR-nn && \
    /tools/miniconda3/bin/conda run -n pytorch pip install -r requirements.txt && \
    /tools/miniconda3/bin/conda run -n pytorch pip install .

# Install PyTorch with GPU support
RUN /tools/miniconda3/bin/conda run -n pytorch conda install \
    pytorch==1.13.1 \
    torchvision==0.14.1 \
    torchaudio==0.13.1 \
    pytorch-cuda=11.6 \
    torchmetrics -c pytorch -c nvidia -y

# Set the HOME variable for the container
ENV HOME=/root

# Install Rust and compile longcallR-dp
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    export PATH="$HOME/.cargo/bin:$PATH" && \
    cd /tools/longcallR-nn/longcallR_dp && \
    cargo build --release

# Set environment variables for runtime
ENV PATH="/tools/miniconda3/bin:/tools/longcallR-nn/longcallR_dp/target/release:${PATH}"
ENV PATH="$HOME/.cargo/bin:${PATH}"

# Clean up unnecessary files to reduce image size
RUN apt-get clean && /tools/miniconda3/bin/conda clean -a -y

# Activate the PyTorch environment
RUN echo "source /tools/miniconda3/bin/activate pytorch" >> ~/.bashrc

# Set the working directory
WORKDIR /work

# Default command
CMD ["/bin/bash"]

