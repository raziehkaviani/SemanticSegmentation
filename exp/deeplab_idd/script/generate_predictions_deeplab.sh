#!/bin/bash
cd /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code && \
python3 exp/deeplab_idd/python/generate_predictions_deeplab.py \
            --data_path /data2/HDD_16TB/IDD/idd_temporal_val_1 \
            --gt_path /data2/HDD_16TB/IDD/IDD_Segmentation \
            --data_list_path /data2/HDD_16TB/IDD/lists/seg/val.txt \
            --save_path /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/result/idd_deeplab/test_results_rgb \
            --scnet_model /home/rkavian1/Desktop/videoSemSeg/BidirCurr_code/saved_model/deeplab_idd/best.pth \
            --num_workers 4


