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
    0: (255, 255, 255),        # Unlabeled
    1: (0, 0, 142),      # Vihicles
    # 2: (157, 234, 50),   # Road
    2: (128, 64, 128),   # Road
    3: (81, 0, 81),      # Ground
}


def main(config,args):
    config.nepoch = args.nepoch
    num_epochs = args.nepoch
    BATCH_SIZE = args.batch
    num_workers = args.nworker
    only_load_model = args.model_only
    start_epoch = 1

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    val_root = args.data.replace('train', 'val')
    valset = V2XSimDataset(dataset_roots=[args.data+'/agent%d'%i for i in range(5)], config=config, split ='val',val=True, com=args.com)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    print("Validation dataset size:", len(valset))


    if args.data.find('upperbound') != -1:
        flag = 'upperbound'
    else:
        if args.com == 'when2com':
            flag = 'when2com'
            if args.warp_flag:
                flag += '_warp'
            if args.inference == 'argmax_test':
                flag = flag.replace('when2com', 'who2com')
        elif args.com == 'V2V':
            flag = 'V2V'
        else:
            flag = 'lowerbound'

    checkpoint = torch.load(args.resume)

    config.flag = flag
    config.com = args.com
    config.inference = args.inference
    config.split = 'test'
    #build model
    if args.com.startswith('when2com') or args.com.startswith('who2com'):
        model = When2Com_UNet(config, in_channels=config.in_channels, n_classes=config.num_class, warp_flag=args.warp_flag)
    elif args.com == 'V2V':
        model = V2V_UNet(config.in_channels, config.num_class)
    else:
        model = UNet(config.in_channels, config.num_class)
    # model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    segmodule = SegModule(model, config, optimizer)
    segmodule.model.load_state_dict(checkpoint['model_state_dict'])
    # ==== eval ====
    segmodule.model.eval()
    compute_iou = ComputeIoU(num_class=config.num_class)  #  num_class
    os.makedirs('./{}_eval'.format(flag), exist_ok=True)

    print('Image will be saved into ./{}_eval'.format(flag))
    for idx, sample in enumerate(tqdm(valloader)):

        t = time.time()
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

        pred, labels = segmodule.step(data, loss=False)
        labels = labels.detach().cpu().numpy().astype(np.int32)
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        compute_iou(pred, labels)

        if args.vis and idx % 50 == 0:  # render segmatic map
            pred_map = np.zeros((256, 256, 3))
            gt_map = np.zeros((256, 256, 3))
            for i in range(256):
                for j in range(256):
                    pred_map[i, j, :] = seg2color[pred.cpu().numpy()[0, i, j]]
                    gt_map[i, j, :] = seg2color[label_one_hot.numpy()[0, i, j]]
            cv2.imwrite('./{}_eval/{}_pred.png'.format(flag, idx), cv2.resize(pred_map, (1024, 1024))[:, :, ::-1])
            cv2.imwrite('./{}_eval/{}_gt.png'.format(flag, idx), cv2.resize(gt_map, (1024, 1024))[:, :, ::-1])
    print('iou:', compute_iou.get_ious())
    print('miou:', compute_iou.get_miou(ignore=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='./dataset/train', type=str, help='The path to the preprocessed sparse BEV training data')
    parser.add_argument('--resume', type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--model_only', action='store_true',help='only load model')
    parser.add_argument('--batch', default=2, type=int, help='Batch size')
    parser.add_argument('--nepoch', default=10, type=int, help='Number of epochs')
    parser.add_argument('--nworker', default=2, type=int, help='Number of workers')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--com', default='', type=str, help='Whether to communicate')
    parser.add_argument('--inference', default='activated')
    parser.add_argument('--warp_flag', action='store_true')
    parser.add_argument('--vis', action='store_true')
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parser.parse_args()
    print(args)
    config = Config('train')
    main(config, args)
