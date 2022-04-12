#!/bin/bash
cd /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code && \
python3 exp/deeplab_idd/python/test_valVideo.py \
            --root_data_path_val /data2/HDD_16TB/IDD/idd_temporal_val_1 \
            --root_gt_path /data2/HDD_16TB/IDD/IDD_Segmentation \
            --test_list_path /data2/HDD_16TB/IDD/lists/seg/val.txt \
            --model_save_path /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/saved_model/pretrained/pretrained_idd_deeplabv3+.pth \


