import os
import sys
import ast
import random
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# from tensorboardX import SummaryWriter

from lib.dataset.idd_bi_strategy2 import idd_video_dataset, idd_video_dataset_PDA
from lib.dataset.utils import runningScore
from lib.model.deeplabv3plus import deeplabv3plus
# from lib.dataset.utils import decode_labels_camvid


def get_arguments():
    parser = argparse.ArgumentParser(description="Train DMNet")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")
    parser.add_argument("--root_data_path", type=str, help="root path to the dataset")
    parser.add_argument("--root_data_path_val", type=str, help="root path to the dataset")
    parser.add_argument("--root_gt_path", type=str, help="root path to the ground truth")
    parser.add_argument("--train_list_path", type=str, help="path to the list of train subset")
    parser.add_argument("--test_list_path", type=str, help="path to the list of test subset")

    ###### training setting ######
    parser.add_argument("--model_name", type=str, help="name for the training model")
    parser.add_argument("--resume", type=ast.literal_eval, default=False, help="resume or not")
    parser.add_argument("--resume_epoch", type=int, help="from which epoch for resume")
    parser.add_argument("--resume_load_path", type=str, help="resume model load path")
    parser.add_argument("--train_load_path", type=str, help="train model load path")
    parser.add_argument("--local_rank", type=int, help="index the replica")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--random_seed", type=int, help="random seed")
    parser.add_argument("--train_flownet", type=ast.literal_eval, default=True, help="trian flownet or not")
    parser.add_argument("--train_power", type=float, help="power value for linear learning rate schedule")
    parser.add_argument("--final_lr", type=float, default=0.00001, help="learning rate in the second stage")
    parser.add_argument("--weight_decay", type=float, help="learning rate")
    parser.add_argument("--train_batch_size", type=int, help="train batch size")
    parser.add_argument("--train_shuffle", type=ast.literal_eval, default=True, help="shuffle or not in training")
    parser.add_argument("--train_num_workers", type=int, default=8, help="num cpu use")
    parser.add_argument("--num_epoch", type=int, default=100, help="num of epoch in training")
    parser.add_argument("--snap_shot", type=int, default=1, help="save model every per snap_shot")
    parser.add_argument("--model_save_path", type=str, help="model save path")

    ###### testing setting ######
    parser.add_argument("--test_batch_size", type=int, default=1, help="batch_size for validation")
    parser.add_argument("--test_shuffle", type=ast.literal_eval, default=False, help="shuffle or not in validation")
    parser.add_argument("--test_num_workers", type=int, default=4, help="num of used cpus in validation")

    ###### tensorboard setting ######
    parser.add_argument("--use_tensorboard", type=ast.literal_eval, default=True, help="use tensorboard or not")
    parser.add_argument("--tblog_dir", type=str, help="log save path")
    parser.add_argument("--tblog_interval", type=int, default=50, help="interval for tensorboard logging")

    return parser.parse_args()



def test():
    args = get_arguments()
    print(args)

    net = deeplabv3plus(n_classes=26)
    old_weight = torch.load(args.model_save_path)
    new_weight = {}
    for k, v in old_weight.items():
        print('-k-', k)
        new_k = k.replace('module.', '')
        # new_k = k.replace('deeplab.', '')
        new_weight[new_k] = v
    net.load_state_dict(new_weight, strict=True)
    net.cuda().eval()
    print('---> loaded model!')

    test_data = idd_video_dataset_PDA(args.root_data_path_val, args.root_gt_path, args.test_list_path)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.test_num_workers)

    miou_cal = runningScore(n_classes=26)
    loss = 0.0
    with torch.no_grad():
         for step, sample in enumerate(test_data_loader):
            img_list, gt_label = sample
            img = img_list[-1]
            print('-deeplab idd-', step, img.shape)
            pred = net(img.cuda())
            pred = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=True)
      
            #loss += F.cross_entropy(pred, gt_label.cuda()).squeeze().item()
            
            out = torch.argmax(pred, dim=1)
            out = out.squeeze().cpu().numpy()
         
            miou_cal.update(gt_label.squeeze().cpu().numpy(), out)

         #   out_rgb = decode_labels_camvid(out)
         #   cv2.imwrite(os.path.join(args.model_save_path, 'out_'+str(step)+'.png'), out_rgb)

         loss /= len(test_data_loader)
    
         print('eval_loss:{}'.format(loss))

         miou, iou = miou_cal.get_scores(return_class=True)
         miou_cal.reset()
         print('eval_miou:{}'.format(miou))
      
         for i in range(len(iou)):
             print(iou[i])

         #out_rgb = decode_labels_camvid(out)
         #cv2.imwrite(os.path.join(args.model_save_path, 'out_'+str(step)+'.png'), out_rgb) 

if __name__ == '__main__':
    test()


