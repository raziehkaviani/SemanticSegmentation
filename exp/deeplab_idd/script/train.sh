#!/bin/bash

cd /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code && \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM  exp/deeplab_idd/python/train.py \
        --exp_name deeplab_idd \
        --root_data_path /data2/HDD_16TB/IDD/IDD_Segmentation \
        --root_gt_path /data2/HDD_16TB/IDD/IDD_Segmentation \
        --train_list_path /data2/HDD_16TB/IDD/lists/seg/train.txt \
        --test_list_path /data2/HDD_16TB/IDD/lists/seg/val.txt \
        --train_load_path /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/saved_model/pretrained/pretrained_cityscapes.pth \
        --lr 1e-4 \
        --random_seed 666 \
        --weight_decay 0 \
        --train_batch_size 2 \
        --train_num_workers 4 \
        --test_batch_size 1 \
        --test_num_workers 4 \
        --num_epoch 200 \
        --snap_shot 2 \
        --model_save_path /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/saved_model/deeplab_idd \
        --tblog_dir /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/tblog1/deeplab_idd


