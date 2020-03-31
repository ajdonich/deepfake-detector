#!/bin/bash

set -e

# Update custon Miniconda environment
SAGEMAKER_DIR=/home/ec2-user/SageMaker/
source "$SAGEMAKER_DIR/opt/miniconda/bin/activate"

conda env remove --name deepfake
conda env create -f "$SAGEMAKER_DIR/kaggle-deepfake-detection/deepfake.yml"

