import torch

import coperception.utils.convolutional_rnn as convrnn
from coperception.models.seg.SegModelBase import SegModelBase
import torch.nn.functional as F


class V2VNet(SegModelBase):
    def __init__(self, n_channels, n_classes, num_agent=5, compress_level=0, only_v2i=False):
        super().__init__(
            n_channels, n_classes, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i
        )
        self.layer_channel = 512
        self.gnn_iter_num = 1
        self.convgru = convrnn.Conv2dGRU(
            in_channels=self.layer_channel * 2,
            out_channels=self.layer_channel,
            kernel_size=3,
            num_layers=1,
            bidirectional=False,
            dilation=1,
            stride=1,
        )

    def forward(self, x, trans_matrices, num_agent_tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # b 512 32 32
        size = (1, 512, 32, 32)

        if self.compress_level > 0:
            x4 = F.relu(self.bn_compress(self.com_compresser(x4)))
            x4 = F.relu(self.bn_decompress(self.com_decompresser(x4)))

        batch_size = x.size(0) // self.num_agent
        feat_list = super().build_feat_list(x4, batch_size)

        local_com_mat = torch.cat(tuple(feat_list), 1)
        local_com_mat_update = torch.cat(tuple(feat_list), 1)

        for b in range(batch_size):
            com_num_agent = num_agent_tensor[b, 0]

            agent_feat_list = list()
            for nb in range(self.num_agent):
                agent_feat_list.append(local_com_mat[b, nb])

            for _ in range(self.gnn_iter_num):
                updated_feats_list = list()

                for i in range(com_num_agent):
                    tg_agent = local_com_mat[b, i]

                    neighbor_feat_list = list()
                    neighbor_feat_list.append(tg_agent)

                    for j in range(com_num_agent):
                        if j != i:
                            if self.only_v2i and i != 0 and j != 0:
                                continue
                            
                            neighbor_feat_list.append(
                                super().feature_transformation(
                                    b,
                                    j,
                                    i,
                                    local_com_mat,
                                    size,
                                    trans_matrices,
                                )
                            )

                    mean_feat = torch.mean(torch.stack(neighbor_feat_list), dim=0)
                    cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
                    cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)
                    updated_feat, _ = self.convgru(cat_feat, None)
                    updated_feat = torch.squeeze(torch.squeeze(updated_feat, 0), 0)
                    updated_feats_list.append(updated_feat)
                agent_feat_list = updated_feats_list
            for k in range(com_num_agent):
                local_com_mat_update[b, k] = agent_feat_list[k]

        feat_mat = super().agents_to_batch(local_com_mat_update)

        x5 = self.down4(feat_mat)
        x = self.up1(x5, feat_mat)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
