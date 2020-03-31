#!/bin/bash

set -e
sudo -u ec2-user -i <<'EOF'

WORKING_DIR=/home/ec2-user/SageMaker/opt/
source "$WORKING_DIR/miniconda/bin/activate"

for env in $WORKING_DIR/miniconda/envs/*; do
  BASENAME=$(basename "$env")
  source activate "$BASENAME"
  python -m ipykernel install --user --name "$BASENAME" --display-name "conda_$BASENAME"
done

EOF

echo "Restarting the Jupyter server.."
restart jupyter-server