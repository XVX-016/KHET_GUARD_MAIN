FROM nvidia/cuda:11.8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.9 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/logs /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV WANDB_DIR=/app/logs
ENV TORCH_HOME=/app/.cache/torch

CMD ["python", "train.py", "--help"]
