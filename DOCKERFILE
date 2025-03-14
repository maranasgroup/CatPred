FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Install Mambaforge
RUN curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh && \
    bash Mambaforge-Linux-x86_64.sh -b -p /opt/conda && \
    rm Mambaforge-Linux-x86_64.sh

# Update PATH
ENV PATH=/opt/conda/bin:$PATH

# Create a new conda environment with Python 3.12.4
RUN conda create -n myenv python=3.12.4 -y

# Activate the environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install necessary packages
RUN conda install pytorch torchvision torchaudio cudatoolkit=12.4 -c pytorch -y

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        psmisc=23.4-2build3 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y --channel=conda-forge --channel=pyg --channel=pytorch --channel=kimlab \
        ca-certificates==2024.12.14 \
        certifi==2024.12.14 \
        faiss-cpu==1.8.0 \
        jupyterlab==4.3.3 \
        openssl==3.4.0 \
        pandas-flavor==0.6.0 \
        pyg==2.6.1 \
        pytorch-scatter==2.1.2 \
        rdkit==2024.09.2 \
        stride==1.6.4 \
        tensorboardx==2.6.2.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    awscli==1.36.23 \
    fair-esm==2.0.0 \
    ipdb==0.13.13 \
    matplotlib==3.10.0 \
    numpy==2.2.0 \
    pandas \
    progres==0.2.7 \
    rotary-embedding-torch==0.6.5 \
    scikit-learn==1.6.0 \
    scipy \
    seaborn==0.13.2 \
    sentencepiece==0.2.0 \
    tqdm==4.67.1 \
    transformers==4.47.1 \
    typed-argument-parser==1.10.1
