#!/bin/bash

# python train.py \
# --dataset GDGRID \
# --dataset_root /home/qingren/Project/Tianchi_dw/Dataset \
# --basenet vgg16_reducedfc.pth \
# --batch_size 4 \
# --start_iter 0 \
# --num_workers 4 \
# --cuda True \
# --lr 1e-3 \
# --momentum 0.9 \
# --weight_decay 5e-4 \
# --gamma 0.1 \
# --save_folder ./weights/ \


# python SSD_train.py \
#   --dataset_root /home/qingren/Project/Tianchi_dw/Dataset \
#   --output_file ./checkpoints/model- \
#   --train \
#   --val \
#   --batch_size 4 \
#   --cuda \
#   --num_workers 4 \
#   --learning_rate 0.001 \
#   --momentum 0.9 \
#   --weight_decay 5e-4 \
#   --lr_step 10 \
#   --lr_step_gamma 0.9 \
#   --log_batch 10 \
#   --val_epoch 1 \
#   --snapshot_epoch 20 \
#   --num_iterations 500 \
# #  --pretrained_model ./checkpoints/model-0.pkl

python SSD_train.py \
  --dataset_root /home/lab/Python_pro/Tianchi/Dataset \
  --output_file ./checkpoints/model- \
  --train \
  --val \
  --batch_size 16 \
  --cuda \
  --num_workers 4 \
  --learning_rate 3e-6 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --lr_step 10 \
  --lr_step_gamma 0.9 \
  --log_batch 1 \
  --val_epoch 1 \
  --snapshot_epoch 5 \
  --num_iterations 500 \
  --pretrained_model ./checkpoints/model-90.pkl
