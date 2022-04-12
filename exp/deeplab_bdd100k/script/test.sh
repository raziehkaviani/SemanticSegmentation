#!/bin/bash

cd /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code && \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM  exp/deeplab_bdd100k/python/test.py \
        --exp_name deeplab_bdd100k \
        --root_data_path /data2/HDD_16TB/BDD100K/bdd100k \
        --root_gt_path /data2/HDD_16TB/BDD100K/bdd100k \
        --train_list_path /data2/HDD_16TB/BDD100K/bdd100k/lists/10k/seg/train_images.txt \
        --test_list_path /data2/HDD_16TB/BDD100K/bdd100k/lists/10k/seg/val_images.txt \
        --train_load_path /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/saved_model/pretrained/pretrained_cityscapes.pth \
        --lr 1e-4 \
        --random_seed 666 \
        --weight_decay 0 \
        --train_batch_size 2 \
        --train_num_workers 4 \
        --test_batch_size 2 \
        --test_num_workers 4 \
        --num_epoch 200 \
        --snap_shot 1 \
        --model_save_path /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/saved_model/deeplab_bdd100k \
        --tblog_dir /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/tblog1/deeplab_bdd100k


