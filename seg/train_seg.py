# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import numpy as np
import time
import sys
import argparse
import os
from shutil import copytree, copy
from utils.SegModule import *
from utils.model import UNet, When2Com_UNet, V2V_UNet
from utils.loss import *
from utils.SegMetrics import ComputeIoU
from data.Dataset_upperbound_seg import V2XSimDataset
from data.config_upperbound import Config

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


seg2color = {
    0: (0, 0, 0),        # Unlabel    -> White
    1: (0, 0, 142),      # Vihicles   -> Blue
    2: (157, 234, 50),   # RoadLine
    3: (81, 0, 81),      # Ground
    4: (128, 64, 128),   # Road
    5: (70, 70, 70),     # Building
    6: (145, 170, 100),  # Terrain
    7: (107, 142, 35),   # Vegetation
}


def main(config,args):
    config.nepoch = args.nepoch
    num_epochs = args.nepoch
    need_log = args.log
    BATCH_SIZE = args.batch
    num_workers = args.nworker
    only_load_model = args.model_only
    start_epoch = 1

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    trainset = V2XSimDataset(dataset_roots=[args.data + '/agent%d'%i for i in range(5)], config=config, split='train', com=args.com)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    print("Training dataset size:", len(trainset))

    val_root = args.data.replace('trian', 'test')
    valset = V2XSimDataset(dataset_roots=[val_root+'/agent%d'%i for i in range(5)], config=config, split ='val',val=True, com=args.com)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    print("Validation dataset size:", len(valset))

    logger_root = args.logpath if args.logpath != '' else 'logs'

    if args.bound == 'lowerbound':
        flag = 'upperbound'
    else:
        if args.com == 'when2com':
            if args.warp_flag:
                flag = 'when2com_warp'
            else:
                flag = 'when2com'
        elif args.com == 'V2V':
            flag = 'V2V'
        else:
            flag = 'lowerbound'
    config.flag = flag
    if args.resume is None:
        model_save_path = check_folder(logger_root)
        # model_save_path = check_folder(os.path.join(model_save_path, 'train_single_seq'))
        model_save_path = check_folder(os.path.join(model_save_path, flag))

        log_file_name = os.path.join(model_save_path, 'log.txt')
        saver = open(log_file_name, "w")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

        # Copy the code files as logs
        # copytree('nuscenes-devkit', os.path.join(model_save_path, 'nuscenes-devkit'), exist_ok=True)
        # copytree('data', os.path.join(model_save_path, 'data'))
        # python_files = [f for f in os.listdir('.') if f.endswith('.py')]
        # for f in python_files:
        #     copy(f, model_save_path, exist_ok=True)
    else:
        model_save_path = args.resume[:args.resume.rfind('/')]
        torch.load(args.resume)  # eg, "logs/train_multi_seq/1234-56-78-11-22-33"

        log_file_name = os.path.join(model_save_path, 'log.txt')
        saver = open(log_file_name, "a")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

    #build model
    if args.com == 'when2com':
        model = When2Com_UNet(config, in_channels=config.in_channels, n_classes=config.num_class, warp_flag=args.warp_flag)
    elif args.com == 'V2V':
        model = V2V_UNet(config.in_channels, config.num_class)
    else:
        model = UNet(config.in_channels, config.num_class)
    # model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)



    config.com = args.com
    segmodule = SegModule(model, config, optimizer)

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        segmodule.model.load_state_dict(checkpoint['model_state_dict'])
        segmodule.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        segmodule.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    for epoch in range(start_epoch, num_epochs + 1):
        lr = segmodule.optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        running_loss_disp = AverageMeter('Total loss', ':.6f')  # for motion prediction error
        segmodule.model.train()

        t = time.time()
        for idx, sample in enumerate(tqdm(trainloader)):

            if args.com:
                padded_voxel_points_list, label_one_hot_list, trans_matrices, target_agent, num_sensor = list(zip(*sample))
            else:
                padded_voxel_points_list, label_one_hot_list = list(zip(*sample))

            padded_voxel_points = torch.cat(tuple(padded_voxel_points_list), 0)
            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            # print('voxel', padded_voxel_points.size())  # batch*agent seq h w z
            # print('label', label_one_hot.size())

            data = {}
            data['bev_seq'] = padded_voxel_points.to(device).float()
            data['labels'] = label_one_hot.to(device)
            if args.com:
                trans_matrices = torch.stack(trans_matrices, 1)
                target_agent = torch.stack(target_agent, 1)
                num_sensor = torch.stack(num_sensor, 1)
                data['trans_matrices'] = trans_matrices
                data['target_agent'] = target_agent
                data['num_sensor'] = num_sensor

            pred, loss = segmodule.step(data)

            running_loss_disp.update(loss)
        print("\nEpoch {}".format(epoch))
        print("Running total loss: {}".format(running_loss_disp.avg))
        segmodule.scheduler.step()
        print("{}\t Takes {} s\n".format(running_loss_disp,str(time.time()-t)))

        if need_log:
            saver.write("{}\n".format(running_loss_disp))
            saver.flush()

        # save model
        if need_log:
            save_dict = {'epoch': epoch,
                         'model_state_dict': segmodule.model.state_dict(),
                         'optimizer_state_dict': segmodule.optimizer.state_dict(),
                         'scheduler_state_dict': segmodule.scheduler.state_dict(),
                         'loss': running_loss_disp.avg}
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))
    if need_log:
        saver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='./dataset/train', type=str, help='The path to the preprocessed sparse BEV training data')
    parser.add_argument('--bound')
    parser.add_argument('--resume', default=None, type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--model_only', action='store_true',help='only load model')
    parser.add_argument('--batch', default=2, type=int, help='Batch size')
    parser.add_argument('--warp_flag', action='store_true')
    parser.add_argument('--augmentation', default=False, help='Whether to use data augmentation')
    parser.add_argument('--nepoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--nworker', default=2, type=int, help='Number of workers')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--log', action='store_true', help='Whether to log')
    parser.add_argument('--logpath', default='./log', help='The path to the output log file')
    parser.add_argument('--com', default='', type=str, help='Whether to communicate')
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parser.parse_args()
    print(args)
    config = Config('train')
    main(config, args)
