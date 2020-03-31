#!/bin/bash

set -e

# Install a custom Miniconda in SageMaker EBS
WORKING_DIR=/home/ec2-user/SageMaker/opt
mkdir -p "$WORKING_DIR"

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$WORKING_DIR/miniconda.sh"
bash "$WORKING_DIR/miniconda.sh" -b -u -p "$WORKING_DIR/miniconda" 
rm -rf "$WORKING_DIR/miniconda.sh"

# Create a custom conda environment
source "$WORKING_DIR/miniconda/bin/activate"
conda env create -f "$WORKING_DIR/../kaggle-deepfake-detection/deepfake.yml"
conda activate deepfake
