import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.model import UNet, STPN, lidar_encoder, lidar_decoder, conv2DBatchNormRelu, Sparsemax
import numpy as np


class ClassificationHead(nn.Module):

    def __init__(self, config):

        super(ClassificationHead, self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, category_num * anchor_num_per_loc, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class MotionStateHead(nn.Module):

    def __init__(self, config):

        super(MotionStateHead, self).__init__()
        category_num = 3  # ignore: 0 static: 1 moving: 2
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, category_num * anchor_num_per_loc, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class FeatEncoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(FeatEncoder, self).__init__()
        self.stpn = STPN(height_feat_size=height_feat_size)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x = self.stpn(bevs)

        return x


class RegressionHead(nn.Module):
    def __init__(self, config):
        super(RegressionHead, self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)
        box_code_size = config.box_code_size

        self.box_prediction = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, anchor_num_per_loc * box_code_size, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        box = self.box_prediction(x)
        return x


class SingleRegressionHead(nn.Module):
    def __init__(self, config):
        super(SingleRegressionHead, self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)
        box_code_size = config.box_code_size
        if config.only_det:
            out_seq_len = 1
        else:
            out_seq_len = config.pred_len

        if config.binary:
            if config.only_det:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
                    nn.Conv2d(channel, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1,
                              padding=0))
            else:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1,
                              padding=0))

    def forward(self, x):
        box = self.box_prediction(x)

        return box


class FaFMGDA(nn.Module):

    def __init__(self, config):
        super(FaFMGDA, self).__init__()
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.motion_state = config.motion_state

        self.box_code_size = config.box_code_size
        self.category_num = config.category_num

        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        # self.RegressionList = nn.ModuleList([RegressionHead for i in range(seq_len)])
        self.regression = SingleRegressionHead(config)
        if self.motion_state:
            self.motion_cls = MotionStateHead(config)

    def forward(self, x):
        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0], -1, self.category_num)

        # Detection head
        loc_preds = self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1, loc_preds.size(1), loc_preds.size(2), self.anchor_num_per_loc, self.out_seq_len,
                                   self.box_code_size)
        # loc_pred (N * T * W * H * loc)

        result = {'loc': loc_preds,
                  'cls': cls_preds}

        # MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0], -1, motion_cat)
            result['state'] = motion_cls_preds

        return result


class FaFNet(nn.Module):

    def __init__(self, config):
        super(FaFNet, self).__init__()
        self.unet = UNet(config.in_channels, config.num_class)

    def forward(self, bevs, maps=None, vis=None):
        bevs = bevs.squeeze(1)
        bevs = bevs.permute(0, 3, 1, 2)  # (Batch, h, w, z) -> (Batch, z, h, w)

        result = self.unet(bevs)

        return result
