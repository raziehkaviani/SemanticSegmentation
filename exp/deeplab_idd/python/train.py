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

from tensorboardX import SummaryWriter

from lib.dataset.idd_imgSeg import idd_video_dataset, idd_video_dataset_PDA
from lib.dataset.utils import runningScore
from lib.model.deeplabv3plus import deeplabv3plus


def get_arguments():
    parser = argparse.ArgumentParser(description="Train DMNet")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")
    parser.add_argument("--root_data_path", type=str, help="root path to the dataset")
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


def make_dirs(args):
    if args.use_tensorboard and not os.path.exists(args.tblog_dir):
        os.makedirs(args.tblog_dir)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)


def train():
    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    args = get_arguments()
    if local_rank == 0:
        print(args)
        make_dirs(args)

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if local_rank == 0:
        print('random seed:{}'.format(random_seed))

    if local_rank == 0 and args.use_tensorboard:
        tblogger = SummaryWriter(args.tblog_dir)

    net = deeplabv3plus(n_classes=26)

    old_weight = torch.load(args.train_load_path)
    new_weight = {}
    for k, v in old_weight.items():
        # print('-', k)
        new_k = k.replace('deeplab.', '')
        new_weight[new_k] = v
    del new_weight['logits.weight']
    del new_weight['logits.bias']
    net.load_state_dict(new_weight, strict=False)#True)
   
    print('---> loaded model!')

    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    start_epoch = 0

    net.cuda()
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[local_rank],
                                              output_device=local_rank,
                                              find_unused_parameters=True)

    train_data = idd_video_dataset(args.root_data_path, args.root_gt_path, args.train_list_path, crop_size=(512, 1024))
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=args.train_batch_size,
                                                    shuffle=False,
                                                    pin_memory=False,
                                                    num_workers=args.train_num_workers,
                                                    drop_last=True,
                                                    sampler=DistributedSampler(train_data,
                                                                               num_replicas=world_size,
                                                                               rank=local_rank,
                                                                               shuffle=True))
    print('--trainload--', len(train_data), len(train_data_loader))
    test_data = idd_video_dataset_PDA(args.root_data_path, args.root_gt_path, args.test_list_path)
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=args.test_batch_size,
                                                   shuffle=args.test_shuffle,
                                                   num_workers=args.test_num_workers)
    print('--testload--', len(test_data), len(test_data_loader))
    deeplab_params = []
    for m in net.modules():
        for p in m.parameters():
            deeplab_params.append(p)
    deeplab_optimizer = optim.Adam(params=deeplab_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
    # deeplab_optimizer = optim.SGD(params=deeplab_params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    deeplab_cost = nn.CrossEntropyLoss(ignore_index=255, reduce=False)

    miou_cal = runningScore(n_classes=26)
    running_loss = 0.0
    current_eval_loss = 100
    itr = start_epoch * len(train_data_loader)
    max_itr = args.num_epoch * len(train_data_loader)
    current_miou = 0

    for epoch in range(start_epoch, args.num_epoch):
               
        # net.train()
        train_data_loader.sampler.set_epoch(epoch)
        for i, data_batch in enumerate(train_data_loader):
            # if i<25:
            img_list, gt_label = data_batch
    
            now_lr = adjust_lr(args, deeplab_optimizer, itr, max_itr, args.lr)

            deeplab_optimizer.zero_grad()
            pred = net(img_list)
            pred =  F.upsample(pred, scale_factor=4, mode='bilinear', align_corners=True)
            loss = deeplab_cost(pred, gt_label.cuda())
            loss = torch.mean(loss)
    
            loss = torch.unsqueeze(loss, 0)
            loss.backward()
            deeplab_optimizer.step()
        
            if local_rank == 0:
                print('epoch:{}/{} batch:{}/{} iter:{} loss:{:05f} lr:{}'.format(epoch, args.num_epoch, i,
                                                                                len(train_data_loader), itr,
                                                                                loss.item(), now_lr))

                if args.use_tensorboard and itr % args.tblog_interval == 0:
                    tblogger.add_scalar('loss', loss.item(), itr)

            itr += 1
        
        dist.barrier()
        
        if (epoch+1) % args.snap_shot == 0:

            net.eval()
            loss = 0.0
            eval_loss = []
            with torch.no_grad():
                for step, sample in enumerate(test_data_loader):
                    img, gt_label = sample
                    print('-deeplab bdd100k-', step)
                    pred = net(img.cuda())
                    pred = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=True)
                    
                    loss += F.cross_entropy(pred, gt_label.cuda(), ignore_index=255).squeeze().item() 

                    out = torch.argmax(pred, dim=1)
                    out = out.cpu().numpy()#.squeeze()
                    # print('-pred-', out.shape, gt_label.shape)
                    miou_cal.update(gt_label.cpu().numpy(), out)

                loss /= len(test_data_loader)
                eval_loss.append(loss)
                print('eval_loss:{}'.format(loss))
            
                miou, iou = miou_cal.get_scores(return_class=True)
                miou_cal.reset()
                print('eval_miou:{}'.format(miou))

                for i in range(len(iou)):
                    print(iou[i])

            with open(os.path.join(args.model_save_path, 'history.log'), 'a') as f:
                f.write('%04d,%.4f,%.4f\n' % (epoch + 1, loss, miou))

            if miou > current_miou:

                save_name = 'best.pth'
                save_path = os.path.join(args.model_save_path, save_name)
                torch.save(net.state_dict(), save_path)
                current_moiu = miou

            dist.barrier()
        
    if local_rank==0:       
        save_name = 'final.pth'
        save_path = os.path.join(args.model_save_path, save_name)
        torch.save(net.state_dict(), save_path)
        print('%s has been saved' % save_path)


def adjust_lr(args, optimizer, itr, max_itr, lr, power=0.9, nbb_mult=10):
#    if itr > max_itr / 2:
#        now_lr = lr / 10
#    else:
#        now_lr = lr
 
    now_lr = lr*((1-float(itr)/max_itr)**(power))

    for group in optimizer.param_groups:
        group['lr'] = now_lr
        #print(group['lr'])
    return now_lr

if __name__ == '__main__':
    train()
    dist.destroy_process_group()



