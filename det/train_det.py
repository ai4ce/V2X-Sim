# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
import matplotlib
matplotlib.use('Agg')
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import sys
import argparse
import os
from shutil import copytree, copy
from utils.model import MotionNet
from utils.DetModel import V2VNet, FaFMIMONet
from utils.DetModule import *
from utils.loss import *
from data.Dataset_com import NuscenesDataset, CarscenesDataset
from data.config_com import Config, ConfigGlobal
from utils.mean_ap import eval_map
import seaborn as sns
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


def main(args):
    config = Config('train', binary=True, only_det=True)
    config_global = ConfigGlobal('train', binary=True, only_det=True)

    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker
    start_epoch = 1
    batch_size = args.batch

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    if args.bound == 'upperbound':
        flag = 'upperbound'
    elif args.bound == 'lowerbound':
        if args.com == 'when2com':
            if args.warp_flag:
                flag = 'when2com_warp'
            else:
                flag = 'when2com'
        elif args.com == 'V2V':
            flag = 'V2V'
        else:
            flag = 'lowerbound'
    else:
        raise ValueError('not implement')
    config.flag = flag

    trainset = CarscenesDataset(dataset_roots=[f'{args.data}/agent{i}' for i in range(5)], config=config, config_global=config_global, split='train')
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    print("Training dataset size:", len(trainset))

    #config.split='val'
    #val_root = args.data.replace('train', 'test')
    #valset = CarscenesDataset(dataset_roots=[f'{val_root}/agent{i}' for i in range(5)], config=config, config_global=config_global, split='val', val=True)
    #valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers)
    #print("Validation dataset size:", len(valset))

    logger_root = args.logpath if args.logpath != '' else 'logs'

    if args.com == '':
        model = FaFNet(config)
    elif args.com == 'when2com':
        # model = PixelwiseWeightedFusionSoftmax(config, layer=args.layer)
        model = FaFMIMONet(config, layer=args.layer, warp_flag=args.warp_flag)
    elif args.com == 'V2V':
        model = V2VNet(config, gnn_iter_times=args.gnn_iter_times, layer=args.layer, layer_channel=256)
    # model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {'cls': SoftmaxFocalClassificationLoss(), 'loc': WeightedSmoothL1LocalizationLoss()}

    fafmodule = FaFModule(model, config, optimizer, criterion)

    if args.resume == '':
        model_save_path = check_folder(logger_root)
        model_save_path = check_folder(os.path.join(model_save_path, flag))

        log_file_name = os.path.join(model_save_path, 'log.txt')
        saver = open(log_file_name, "w")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()
    else:
        model_save_path = args.resume[:args.resume.rfind('/')]

        log_file_name = os.path.join(model_save_path, 'log.txt')
        saver = open(log_file_name, "a")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        fafmodule.model.load_state_dict(checkpoint['model_state_dict'])
        fafmodule.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        fafmodule.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    for epoch in range(start_epoch, num_epochs + 1):
        lr = fafmodule.optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        running_loss_disp = AverageMeter('Total loss', ':.6f')  # for motion prediction error
        running_loss_class = AverageMeter('classification Loss', ':.6f')  # for cell classification error
        running_loss_loc = AverageMeter('Localization Loss', ':.6f')  # for state estimation error

        fafmodule.model.train()

        t = tqdm(trainloader)
        for sample in t:
            padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list,\
                target_agent_id_list, num_agent_list, trans_matrices_list = zip(*sample)

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_agent = torch.stack(tuple(num_agent_list), 1)

            if flag == 'upperbound':
                padded_voxel_point = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
            else:
                padded_voxel_point = torch.cat(tuple(padded_voxel_point_list), 0)
            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)

            # print('padded_voxel_points', padded_voxel_points.size())
            # print('label_one_hot', label_one_hot.size())
            # print('reg_target', reg_target.size())
            # print('reg_loss_mask', reg_loss_mask.size())
            # print('anchors_map', anchors_map.size())
            # print('vis_maps', vis_maps.size())
            # print('trans_matrices', trans_matrices.size())
            # print('target_agent_ids', target_agent_ids.size())
            # print('num_agent', num_agent.size())

            data = {}
            data['bev_seq'] = padded_voxel_point.to(device)  # [batch, agent, 1, 256, 256, 13] [batch*agent, 1, 256, 256, 13]
            data['labels'] = label_one_hot.to(device)
            data['reg_targets'] = reg_target.to(device)
            data['anchors'] = anchors_map.to(device)
            data['reg_loss_mask'] = reg_loss_mask.to(device).type(dtype=torch.bool)
            data['vis_maps'] = vis_maps.to(device)

            data['target_agent_ids'] = target_agent_id.to(device)
            data['num_agent'] = num_agent.to(device)
            data['trans_matrices'] = trans_matrices  # [4, 5, 5, 4, 4] [4, 5, 5, 4, 4]

            loss, cls_loss, loc_loss = fafmodule.step(data, batch_size)
            running_loss_disp.update(loss)
            running_loss_class.update(cls_loss)
            running_loss_loc.update(loc_loss)

            if np.isnan(loss) or np.isnan(cls_loss) or np.isnan(loc_loss):
                print(f'Epoch {epoch}, loss is nan: {loss}, {cls_loss} {loc_loss}')
                sys.exit();

            t.set_description("Epoch {},     lr {}".format(epoch, lr))
            t.set_postfix(cls_loss=running_loss_class.avg, loc_loss=running_loss_loc.avg)

        fafmodule.scheduler.step()

        # save model
        if need_log:
            saver.write("{}\t{}\t{}\n".format(running_loss_disp, running_loss_class, running_loss_loc))
            saver.flush()
            if config.MGDA:
                save_dict = {'epoch': epoch,
                             'encoder_state_dict': fafmodule.encoder.state_dict(),
                             'optimizer_encoder_state_dict': fafmodule.optimizer_encoder.state_dict(),
                             'scheduler_encoder_state_dict': fafmodule.scheduler_encoder.state_dict(),
                             'head_state_dict': fafmodule.head.state_dict(),
                             'optimizer_head_state_dict': fafmodule.optimizer_head.state_dict(),
                             'scheduler_head_state_dict': fafmodule.scheduler_head.state_dict(),
                             'loss': running_loss_disp.avg}
            else:
                save_dict = {'epoch': epoch,
                             'model_state_dict': fafmodule.model.state_dict(),
                             'optimizer_state_dict': fafmodule.optimizer.state_dict(),
                             'scheduler_state_dict': fafmodule.scheduler.state_dict(),
                             'loss': running_loss_disp.avg}
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))

        if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
    parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--pose_error_trans', default=1.0, type=float, help='Variance of pose error in translation')
    # the default mode is single layer communication, set the layer num
    parser.add_argument('--layer', default=3, type=int, help='Communicate which layer in the single layer com mode')
    parser.add_argument('--warp_flag', action='store_true')
    parser.add_argument('--batch', default=4, type=int, help='Batch size')
    parser.add_argument('--nepoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--nworker', default=2, type=int, help='Number of workers')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--log', action='store_true', help='Whether to log')
    parser.add_argument('--logpath', default='./log', help='The path to the output log file')
    parser.add_argument('--mode', default=None, help='Train/Val mode')
    parser.add_argument('--visualization', default=True, help='Visualize validation result')
    parser.add_argument('--gnn_iter_times', default=1, type=int, help='Number of message passing')
    parser.add_argument('--com', default='', type=str, help='Whether to communicate')
    parser.add_argument('--bound', type=str)
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    print(args)
    main(args)
