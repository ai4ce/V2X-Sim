# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
import torch.nn.functional as F
import torch.nn as nn
import torch
import convolutional_rnn as convrnn
import copy
import numpy as np

##################
#      Unet      # ref: https://github.com/milesial/Pytorch-UNet
##################
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


##################
#      Unet      #
##################

class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()
        self.conv1 = nn.Conv2d(channel*2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(8)

        self.conv4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        return x


class policy_net4(nn.Module):
    def __init__(self, in_channels=13, input_feat_sz=32):
        super(policy_net4, self).__init__()
        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(256 * feat_map_sz * feat_map_sz)
        # self.lidar_encoder = lidar_encoder(height_feat_size=in_channels)

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # Encoder
        # down 1 
        self.conv1 = conv2DBatchNormRelu(512, 512, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(512, 256, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

        # down 2
        self.conv4 = conv2DBatchNormRelu(256, 256, k_size=3, stride=1, padding=1)
        self.conv5 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

    def forward(self, features_map):
        # _, _, _, _, outputs1 = self.lidar_encoder(features_map)
        outputs1 = self.down3(self.down2(self.down1(self.inc(features_map))))
        outputs = self.conv1(outputs1)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs

class km_generator(nn.Module):
    def __init__(self, out_size=128, input_feat_sz=32):
        super(km_generator, self).__init__()
        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(256 * feat_map_sz * feat_map_sz)
        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256), #            
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), #             
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size)) #            

    def forward(self, features_map):
        outputs = self.fc(features_map.view(-1, self.n_feat))
        return outputs

# MIMO (non warp)
class MIMOGeneralDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, query_size, key_size, warp_flag, attn_dropout=0.1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(query_size, key_size)
        self.warp_flag = warp_flag
        print('Msg size: ',query_size,'  Key size: ', key_size)

    def forward(self, qu, k, v, sparse=True):
        # qu (batch,5,32)
        # k (batch,5,1024)
        # v (batch,5,channel,size,size)
        query = self.linear(qu)  # (batch,5,key_size)

        # normalization
        # query_norm = query.norm(p=2,dim=2).unsqueeze(2).expand_as(query)
        # query = query.div(query_norm + 1e-9)

        # k_norm = k.norm(p=2,dim=2).unsqueeze(2).expand_as(k)
        # k = k.div(k_norm + 1e-9)
        # generate the
        attn_orig = torch.bmm(k, query.transpose(2, 1))  # (batch,5,5)  column: differnt keys and the same query

        # scaling [not sure]
        # scaling = torch.sqrt(torch.tensor(k.shape[2],dtype=torch.float32)).cuda()
        # attn_orig = attn_orig/ scaling # (batch,5,5)  column: differnt keys and the same query

        attn_orig_softmax = self.softmax(attn_orig)  # (batch,5,5)

        attn_shape = attn_orig_softmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        attn_orig_softmax_exp = attn_orig_softmax.view(bats, key_num, query_num, 1, 1, 1)

        if self.warp_flag==1:
            v_exp = v
        else:
            v_exp = torch.unsqueeze(v, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = attn_orig_softmax_exp * v_exp  # (batch,5,channel,size,size)
        output_sum = output.sum(1)  # (batch,1,channel,size,size)

        return output_sum, attn_orig_softmax

class When2Com_UNet(UNet):
    def __init__(self, config, n_classes=21, in_channels=13, feat_channel=512, feat_squeezer=-1, attention='additive',
                 has_query=True, sparse=False, layer=3, warp_flag=1, image_size=512,
                 shared_img_encoder='unified', key_size=1024, query_size=32):
        super(When2Com_UNet, self).__init__(in_channels, n_classes)
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        # self.classification = ClassificationHead(config)
        # self.regression = SingleRegressionHead(config)

        self.sparse = sparse
        # self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.key_size = key_size
        self.query_size = query_size
        self.shared_img_encoder = shared_img_encoder
        self.has_query = has_query
        self.warp_flag = warp_flag
        self.layer = layer

        self.key_net = km_generator(out_size=self.key_size, input_feat_sz=image_size / 32)
        self.attention_net = MIMOGeneralDotProductAttention(self.query_size, self.key_size, self.warp_flag)
        # # Message generator
        self.query_key_net = policy_net4(in_channels=in_channels)
        if self.has_query:
            self.query_net = km_generator(out_size=self.query_size, input_feat_sz=image_size / 32)

        # Detection decoder
        # self.decoder = lidar_decoder(height_feat_size=in_channels)

        # List the parameters of each modules
        self.attention_paras = list(self.attention_net.parameters())
        #if self.shared_img_encoder == 'unified':
        #    self.img_net_paras = list(self.u_encoder.parameters()) + list(self.decoder.parameters())

        self.policy_net_paras = list(self.query_key_net.parameters()) + list(
            self.key_net.parameters()) + self.attention_paras
        if self.has_query:
            self.policy_net_paras = self.policy_net_paras + list(self.query_net.parameters())

        #self.all_paras = self.img_net_paras + self.policy_net_paras

        #if self.motion_state:
        #    self.motion_cls = MotionStateHead(config)

    def argmax_select(self, warp_flag, val_mat, prob_action):
        # v(batch, query_num, channel, size, size)
        cls_num = prob_action.shape[1]

        coef_argmax = F.one_hot(prob_action.max(dim=1)[1],  num_classes=cls_num).type(torch.cuda.FloatTensor)
        coef_argmax = coef_argmax.transpose(1, 2)
        attn_shape = coef_argmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        coef_argmax_exp = coef_argmax.view(bats, key_num, query_num, 1, 1, 1)


        if warp_flag==1:
            v_exp = val_mat
        else:
            v_exp = torch.unsqueeze(val_mat, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)


        output = coef_argmax_exp * v_exp  # (batch,4,channel,size,size)
        feat_argmax = output.sum(1)  # (batch,1,channel,size,size)


        # compute connect
        count_coef = copy.deepcopy(coef_argmax)
        ind = np.diag_indices(self.agent_num)
        count_coef[:, ind[0], ind[1]] = 0
        num_connect = torch.nonzero(count_coef).shape[0] / (self.agent_num * count_coef.shape[0])

        return feat_argmax, coef_argmax, num_connect

    def activated_select(self, warp_flag, val_mat, prob_action, thres=0.2):

        coef_act = torch.mul(prob_action, (prob_action > thres).float())
        attn_shape = coef_act.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        coef_act_exp = coef_act.view(bats, key_num, query_num, 1, 1, 1)

        if warp_flag==1:
            v_exp = val_mat
        else:
            v_exp = torch.unsqueeze(val_mat, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = coef_act_exp * v_exp  # (batch,4,channel,size,size)
        feat_act = output.sum(1)  # (batch,1,channel,size,size)

        # compute connect
        count_coef = coef_act.clone()
        ind = np.diag_indices(self.agent_num)
        count_coef[:, ind[0], ind[1]] = 0
        num_connect = torch.nonzero(count_coef).shape[0] / (self.agent_num * count_coef.shape[0])
        return feat_act, coef_act, num_connect

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, maps=None, vis=None, training=True, MO_flag=True, inference='activated', batch_size=1):
        device = bevs.device
        x1 = self.inc(bevs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # b 512 32 32
        size = (1, 512, 32, 32)
        batch_size = bevs.size(0) // self.agent_num
        val_mat = torch.zeros(batch_size, 5, 5, 512, 32, 32).to(device)

        # get feat maps for each agent
        feat_map = {}
        feat_list = []
        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(x4[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
         generate value matrix for each agent, Yiming, 2021.4.22
         
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if self.warp_flag==1:
            local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
            for b in range(batch_size):
                num_agent = num_agent_tensor[b, 0]
                for i in range(num_agent):
                    tg_agent = local_com_mat[b, i]
                    all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                    for j in range(num_agent):
                        if j==i:
                            val_mat[b, i, j] = tg_agent
                        else:
                            nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                            nb_warp = all_warp[j] # [4 4]
                            # normalize the translation vector
                            x_trans = (4*nb_warp[0, 3])/128
                            y_trans = -(4*nb_warp[1, 3])/128

                            theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                            theta_rot = torch.unsqueeze(theta_rot, 0)
                            grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                            theta_trans = torch.unsqueeze(theta_trans, 0)
                            grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                            #first rotate the feature map, then translate it
                            warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                            warp_feat = torch.squeeze(warp_feat_trans)

                            val_mat[b, i, j] = warp_feat
        else:
            val_mat = torch.cat(tuple(feat_list), 1)

        # pass feature maps through key and query generator
        query_key_maps = self.query_key_net(bevs)
        keys = self.key_net(query_key_maps)
        
        if self.has_query:
            querys = self.query_net(query_key_maps)
        # get key and query
        key = {}
        query = {}
        key_list = []
        query_list = []

        for i in range(self.agent_num):
            key[i] = torch.unsqueeze(keys[batch_size * i:batch_size * (i + 1)], 1)
            key_list.append(key[i])
            if self.has_query:
                query[i] = torch.unsqueeze(querys[batch_size * i:batch_size * (i + 1)], 1)
            else:
                query[i] = torch.ones(batch_size, 1, self.query_size).to('cuda')
            query_list.append(query[i])

        key_mat = torch.cat(tuple(key_list), 1)
        query_mat = torch.cat(tuple(query_list), 1)
        if MO_flag:
            query_mat = query_mat
        else:
            query_mat = torch.unsqueeze(query_mat[:,0,:],1)

        feat_fuse, prob_action = self.attention_net(query_mat, key_mat, val_mat, sparse=self.sparse)
        #print(query_mat.shape, key_mat.shape, val_mat.shape, feat_fuse.shape)
        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(feat_fuse)

        # not related to how we combine the feature (prefer to use the agnets' own frames: to reduce the bandwidth)
        small_bis = torch.eye(prob_action.shape[1])*0.001
        small_bis = small_bis.reshape((1, prob_action.shape[1], prob_action.shape[2]))
        small_bis = small_bis.repeat(prob_action.shape[0], 1, 1).cuda()
        prob_action = prob_action + small_bis

        if training:
            action = torch.argmax(prob_action, dim=1)
            num_connect = self.agent_num - 1
            x5 = self.down4(feat_fuse_mat)
            x = self.up1(x5, feat_fuse_mat)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:
            if inference == 'softmax':
                action = torch.argmax(prob_action, dim=1)
                num_connect = self.agent_num - 1
                x5 = self.down4(feat_fuse_mat)
                x = self.up1(x5, feat_fuse_mat)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)
            elif inference == 'argmax_test':
                # print('argmax_test')
                feat_argmax, connect_mat, num_connect = self.argmax_select(self.warp_flag, val_mat, prob_action)
                feat_argmax_mat = self.agents2batch(feat_argmax)  # (batchsize*agent_num, channel, size, size)
                feat_argmax_mat = feat_argmax_mat.detach()
                # pred_argmax = self.decoder(x, x_1, x_2, feat_argmax_mat, x_4, batch_size)

                x5 = self.down4(feat_argmax_mat)
                x = self.up1(x5, feat_argmax_mat)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)
                # action = torch.argmax(connect_mat, dim=1)
                # return pred_argmax, prob_action, action, num_connect
                # x=pred_argmax
            elif inference == 'activated':
                # print('activated')
                feat_act, connect_mat, num_connect = self.activated_select(self.warp_flag, val_mat, prob_action)
                feat_act_mat = self.agents2batch(feat_act)  # (batchsize*agent_num, channel, size, size)
                feat_act_mat = feat_act_mat.detach()
                #if self.layer ==4:
                #    pred_act = self.decoder(x, x_1, x_2, x_3, feat_act_mat,batch_size)
                #elif self.layer == 3:
                #    pred_act = self.decoder(x, x_1, x_2, feat_act_mat, x_4, batch_size)
                #elif self.layer == 2:
                #    pred_act = self.decoder(x, x_1, feat_act_mat, x_3, x_4, batch_size)
                x5 = self.down4(feat_act_mat)
                x = self.up1(x5, feat_act_mat)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)
            else:
                raise ValueError('Incorrect inference mode')
        logits = self.outc(x)
        return logits


class When2Com_UNet_replicated(UNet):
    def __init__(self, n_channels, n_classes, fusion_layer=3):
        super(When2Com_UNet, self).__init__(n_channels, n_classes)
        self.num_agent = 5
        self.fusioner = PixelWeightedFusionSoftmax(512)

    def forward(self, x, trans_matrices, num_agent_tensor):
        device = x.device
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # b 512 32 32
        size = (1, 512, 32, 32)

        feat_map = {}
        feat_list = []
        batch_size = x.size(0) // self.num_agent
        for i in range(self.num_agent):
            feat_map[i] = torch.unsqueeze(x4[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])
        local_com_mat = torch.cat(tuple(feat_list), 1)
        local_com_mat_update = torch.cat(tuple(feat_list), 1)
        save_agent_weight_list = list()

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i]

                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)
                
                for j in range(num_agent):
                    nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)
                    nb_warp = all_warp[j]
                    x_trans = 4 * nb_warp[0, 3] / 128
                    y_trans = -4 * nb_warp[1, 3] / 128
                    theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                    theta_rot = torch.unsqueeze(theta_rot, 0)
                    grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))
                    theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                    theta_trans = torch.unsqueeze(theta_trans, 0)
                    grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))
                    warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                    warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                    warp_feat = torch.squeeze(warp_feat_trans)
                    neighbor_feat_list.append(warp_feat)
                tmp_agent_weight_list =list()
                sum_weight = 0
                for k in range(num_agent):
                    cat_feat = torch.cat([tg_agent, neighbor_feat_list[k]], dim=0)
                    cat_feat = cat_feat.unsqueeze(0)
                    AgentWeight = torch.squeeze(self.fusioner(cat_feat))
                    tmp_agent_weight_list.append(torch.exp(AgentWeight))
                    sum_weight = sum_weight + torch.exp(AgentWeight)

                agent_weight_list = list()
                for k in range(num_agent):
                    AgentWeight = torch.div(tmp_agent_weight_list[k], sum_weight)
                    AgentWeight.expand([256, -1, -1])
                    agent_weight_list.append(AgentWeight)

                agent_wise_weight_feat = 0
                for k in range(num_agent):
                    agent_wise_weight_feat = agent_wise_weight_feat + agent_weight_list[k]*neighbor_feat_list[k]

            local_com_mat_update[b, i] = agent_wise_weight_feat

        feat_list = []
        for i in range(self.num_agent):
            feat_list.append(local_com_mat_update[:, i, :, :, :])
        feat_mat = torch.cat(feat_list, 0)

        x5 = self.down4(feat_mat)
        x = self.up1(x5, feat_mat)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class V2V_UNet(UNet):
    def __init__(self, n_channels, n_classes, fusion_layer=3):
        super(V2V_UNet, self).__init__(n_channels, n_classes)
        self.num_agent = 5
        self.layer_channel = 512
        self.gnn_iter_num = 1
        self.convgru = convrnn.Conv2dGRU(in_channels=self.layer_channel * 2,
                                         out_channels=self.layer_channel,
                                         kernel_size=3,
                                         num_layers=1,
                                         bidirectional=False,
                                         dilation=1,
                                         stride=1)

    def forward(self, x, trans_matrices, num_agent_tensor):
        device = x.device
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # b 512 32 32
        size = (1, 512, 32, 32)

        feat_map = {}
        feat_list = []
        batch_size = x.size(0) // self.num_agent
        for i in range(self.num_agent):
            feat_map[i] = torch.unsqueeze(x4[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])
        local_com_mat = torch.cat(tuple(feat_list), 1)
        local_com_mat_update = torch.cat(tuple(feat_list), 1)
        save_agent_weight_list = list()

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]

            agent_feat_list = list()
            for nb in range(self.num_agent):
                agent_feat_list.append(local_com_mat[b, nb])

            for _ in range(self.gnn_iter_num):
                updated_feats_list = list()
                
                for i in range(num_agent):
                    tg_agent = local_com_mat[b, i]
                    all_warp = trans_matrices[b, i]

                    neighbor_feat_list = list()
                    neighbor_feat_list.append(tg_agent)
                
                    for j in range(num_agent):
                        if j!= i:
                            nb_agent = torch.unsqueeze(agent_feat_list[j], 0)
                            nb_warp = all_warp[j]
                            x_trans = 4 * nb_warp[0, 3] / 128
                            y_trans = -4 * nb_warp[1, 3] / 128

                            theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                            theta_rot = torch.unsqueeze(theta_rot, 0)
                            grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))

                            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                            theta_trans = torch.unsqueeze(theta_trans, 0)
                            grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))

                            warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                            warp_feat = torch.squeeze(warp_feat_trans)

                            neighbor_feat_list.append(warp_feat)
                    mean_feat = torch.mean(torch.stack(neighbor_feat_list), dim=0)
                    cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
                    cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)
                    updated_feat, _ = self.convgru(cat_feat, None)
                    updated_feat = torch.squeeze(torch.squeeze(updated_feat, 0), 0)
                    updated_feats_list.append(updated_feat)
                agent_feat_list = updated_feats_list
            for k in range(num_agent):
                local_com_mat_update[b, k] = agent_feat_list[k]

        feat_list = []
        for i in range(self.num_agent):
            feat_list.append(local_com_mat_update[:, i, :, :, :])
        feat_mat = torch.cat(feat_list, 0)

        x5 = self.down4(feat_mat)
        x = self.up1(x5, feat_mat)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x


class STPN(nn.Module):
    def __init__(self, height_feat_size=13):
        super(STPN, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        #        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        #        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv9_1 = nn.Conv2d(32, 6, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

        self.bn9_1 = nn.BatchNorm2d(6)
        self.bn9_2 = nn.BatchNorm2d(6)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        x_8 = F.relu(self.bn8_2(self.conv8_2(x_8)))

        x_9 = F.relu(self.bn9_1(self.conv9_1(x_8)))
        res_x = F.relu(self.bn9_2(self.conv9_2(x_9)))

        return res_x


'''''''''''''''''''''
Added by Yiming
'''''''''''''''''''''


class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.range(start=1, end=number_of_logits, device=input.device).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


class lidar_encoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(lidar_encoder, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        #        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        #        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        return x, x_1, x_2, x_3, x_4


class lidar_decoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(lidar_decoder, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        #        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        #        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x, x_1, x_2, x_3, x_4, batch):
        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        # x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        # x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x
