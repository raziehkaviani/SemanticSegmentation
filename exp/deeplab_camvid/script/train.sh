#!/bin/bash

cd /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code && \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM  exp/deeplab_camvid/python/train.py \
        --exp_name deeplab_camvid \
        --root_data_path /data/datasets/camvid \
        --root_gt_path /data/datasets/camvid \
        --train_list_path /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/data/list/camvid/trainval.txt \
        --test_list_path /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/data/list/camvid/test.txt \
        --train_load_path /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/saved_model/pretrained/pretrained_camvid.pth \
        --lr 1e-4 \
        --random_seed 666 \
        --weight_decay 0 \
        --train_batch_size 2 \
        --train_num_workers 4 \
        --test_batch_size 2 \
        --test_num_workers 1 \
        --num_epoch 500 \
        --snap_shot 1 \
        --model_save_path /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/saved_model/deeplab_camvid \
        --tblog_dir /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/tblog1/deeplab_camvid


