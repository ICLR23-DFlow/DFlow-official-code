#!/bin/bash
mkdir -p checkpoints

python -u train.py --name DFlow --stage DFlow --gpus 0 1 --num_steps 100000 --batch_size 12 --lr 0.0004 --image_size 368 496 --wdecay 0.0001

