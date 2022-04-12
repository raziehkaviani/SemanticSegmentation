#!/bin/bash
cd /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code && \
python3 exp/deeplab_idd/python/generate_predictions_hrnet.py \
            --data_path /data2/HDD_16TB/IDD/idd_temporal_val_1 \
            --gt_path /data2/HDD_16TB/IDD/IDD_Segmentation \
            --data_list_path /data2/HDD_16TB/IDD/lists/seg/val.txt \
            --save_path /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/result/idd_hrnet/test_results_rgb \
            --scnet_model /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/saved_model/pretrained/pretrained_idd_hrnetv2.pth \
            --num_workers 4


