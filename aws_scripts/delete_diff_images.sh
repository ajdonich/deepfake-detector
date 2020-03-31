#!/bin/bash

set -e

DATA_DIR=/home/ec2-user/SageMaker/ebs/deepfake-detect-datalake

echo "The following frame directories exist:"
for framepath in $DATA_DIR/dfdc_frames_part_*; do
    FRAMEDIR=$(basename "$framepath")
    echo "  $FRAMEDIR"
done

echo
read -p "Are you sure you want to delete all of them [[n]/y]? " -r
echo    # move to a new line

if [[ $REPLY =~ ^[Yy]$ ]]
then
    for framepath in $DATA_DIR/dfdc_frames_part_*; do
        FRAMEDIR=$(basename "$framepath")
        echo "Deleting:  $FRAMEDIR"
        rm -rf "$DATA_DIR/$FRAMEDIR"
    done
fi
