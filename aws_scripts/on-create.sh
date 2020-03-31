#!/bin/bash

set -e
sudo -u ec2-user -i <<'EOF'

# Install a separate conda installation via Miniconda
SAGEMAKER_DIR=/home/ec2-user/SageMaker
mkdir -p "$SAGEMAKER_DIR/opt"

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$SAGEMAKER_DIR/opt/miniconda.sh"
bash "$SAGEMAKER_DIR/opt/miniconda.sh" -b -u -p "$SAGEMAKER_DIR/opt/miniconda" 
rm -rf "$SAGEMAKER_DIR/opt/miniconda.sh"

# Create a custom conda environment
source "$SAGEMAKER_DIR/opt/miniconda/bin/activate"
conda env create -f "$SAGEMAKER_DIR/kaggle-deepfake-detection/deepfake.yml"
conda activate deepfake

EOF