import os
import sys
import cv2
import argparse
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# from lib.model.scnet_biCurr_attn_2_idd import SCNet
from lib.model.deeplabv3plus import deeplabv3plus
#from lib.dataset.cityscapes import cityscapes_video_dataset_PDA
from lib.dataset.utils import decode_labels_idd
from lib.dataset.utils import transform_im, randomcrop

from lib.visualize_flow import flow2rgb


def get_arguments():
    parser = argparse.ArgumentParser(description="Test the SCNet")
    ###### general setting ######
    parser.add_argument("--data_list_path", type=str, help="path to the data list")
    parser.add_argument("--data_path", type=str, help="path to the data")
    parser.add_argument("--gt_path", type=str, help="path to the ground truth")
    parser.add_argument("--save_path", type=str, help="path to save results")
    parser.add_argument("--scnet_model", type=str, help="path to the trained SCNet model")
    parser.add_argument("--train_load_path", type=str, help="path to the pretrained models")

    ###### inference setting ######
    parser.add_argument("--num_workers", type=int, help="num of cpus used")

    return parser.parse_args()


def length_sq(x):
    return torch.sum(x**2, 0, keepdims=True)

def test():
    args = get_arguments()
    print(args)

    # net = SCNet(n_classes=11)
    net = deeplabv3plus(n_classes=26)
    
    old_weight = torch.load(args.scnet_model)
    new_weight = {}
    for k, v in old_weight.items():
        print('-', k)
        new_k = k.replace('module.', '')
        new_weight[new_k] = v
    net.load_state_dict(new_weight, strict=True)
    
    net.cuda().eval()

    # deeplab = net.deeplab
    # flownet = net.flownet
    # cfnet = net.cfnet
    # dmnet = net.dmnet
    # warpnet = net.warpnet
  
    #test_data = cityscapes_video_dataset_PDA(args.data_path, args.gt_path, args.data_list_path)
    #test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    DP = 5+1  # 2 ... 10

    #if not os.path.exists(args.save_path):
    #    os.makedirs(args.save_path)
    
    # I = sorted(glob.glob(os.path.join(args.gt_path, "*", "*.png")))
    with open(args.data_list_path, 'r') as f:
        lines = f.readlines()
   
    with torch.no_grad():
        for i, line in enumerate(lines):
            img_name, gt_name = line.split()
            parts = img_name.split("/")
            seq_path = os.path.join(args.data_path, '{:05d}'.format(int(parts[-2])), parts[-1][:-4])
            print('-seq_path-', seq_path)
            if os.path.exists(seq_path):
                seq_list = glob.glob(os.path.join(seq_path, '*.jpeg'))
                img_2_name = os.path.join(seq_path, parts[-1][:-4] + '.jpeg')
                print('-parts-', parts, seq_path, img_2_name)
                seq_list.remove(img_2_name)
                seq_list.sort()
                img_gt_id = int(seq_list[0][-12:-5]) + 15
                start_frame = int(seq_list[0][-12:-5])

                frames_per_video = 30
                
                if not os.path.exists(os.path.join(args.save_path, '{:05d}'.format(int(parts[-2])), parts[-1][:-4])):
                    os.makedirs(os.path.join(args.save_path, '{:05d}'.format(int(parts[-2])), parts[-1][:-4]))

                result_list = []
                for i in range(0, frames_per_video):
                    
                    key_frame_f = start_frame + i
                    if key_frame_f==img_gt_id:
                        path = img_2_name
                    else:
                        path = os.path.join(seq_path, '{:07d}.jpeg'.format(key_frame_f))
                    print('-path-', path)
                    img = transform_im(cv2.imread(path))
                    img = torch.from_numpy(np.expand_dims(img, axis=0))
                    img_f = img.cuda()
                    
                    feat_f = net(img_f)
                    feat_up = F.interpolate(feat_f, scale_factor=4, mode='bilinear', align_corners=True)
                    feat_pred = pred2im(feat_up)
                    # print('-out-',os.path.join(args.save_path, '{:05d}'.format(int(parts[-2])), parts[-1][:-4], '{:07d}'.format(key_frame_f)+ '.png'))
                    cv2.imwrite(os.path.join(args.save_path, '{:05d}'.format(int(parts[-2])), parts[-1][:-4], '{:07d}'.format(key_frame_f)+ '.png'), feat_pred)
                    result_list.append(feat_pred) 



def pred2im(pred):
    pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    pred = decode_labels_idd(pred)
    return pred


if __name__ == '__main__':
    test()



