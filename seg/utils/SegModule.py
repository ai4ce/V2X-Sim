import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.SegModel import FaFNet, FeatEncoder, FaFMGDA
from utils.detection_util import *
from utils.min_norm_solvers import MinNormSolver
import numpy


class SegModule(object):
    def __init__(self, model, config, optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.nepoch)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()

        self.com = config.com

    def resume(self, path):
        def map_func(storage, location):
            return storage.cuda()

        if os.path.isfile(path):
            if rank == 0:
                print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path, map_location=map_func)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

            ckpt_keys = set(checkpoint['state_dict'].keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print('caution: missing keys from checkpoint {}: {}'.format(path, k))
        else:
            print("=> no checkpoint found at '{}'".format(path))

    def step(self, data, loss=True):
        bev = data['bev_seq']
        labels = data['labels']
        self.optimizer.zero_grad()
        bev = bev.permute(0, 3, 1, 2).contiguous()
        if not self.com:
            filtered_bev = []
            filtered_label = []
            for i in range(bev.size(0)):
                if torch.sum(bev[i]) > 1e-4:
                    filtered_bev.append(bev[i])
                    filtered_label.append(labels[i])
            bev = torch.stack(filtered_bev, 0)
            labels = torch.stack(filtered_label, 0)

        if self.com:
            if self.config.flag.startswith('when2com') or self.config.flag.startswith('who2com'):
                if self.config.split == 'train':
                    pred = self.model(bev, data['trans_matrices'], data['num_sensor'], training=True)
                else:
                    pred = self.model(bev, data['trans_matrices'], data['num_sensor'], inference=self.config.inference, training=False)
            else:
                pred = self.model(bev, data['trans_matrices'], data['num_sensor'])
        else:
            pred = self.model(bev)
        if self.com:
            filtered_pred = []
            filtered_label = []
            for i in range(bev.size(0)):
                if torch.sum(bev[i]) > 1e-4:
                    filtered_pred.append(pred[i])
                    filtered_label.append(labels[i])
            pred = torch.stack(filtered_pred, 0)
            labels = torch.stack(filtered_label, 0)
        if not loss:
            return pred, labels

        loss = self.criterion(pred, labels.long())
        if isinstance(self.criterion, nn.DataParallel):
            loss = loss.mean()

        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while training')

        loss.backward()
        self.optimizer.step()

        return pred, loss_data
