import torch
from utils.models.base.FusionBase import FusionBase


class SumFusion(FusionBase):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True, num_agent=5):
        super(SumFusion, self).__init__(config, layer, in_channels, kd_flag, num_agent)

    def fusion(self):
        return torch.sum(torch.stack(self.neighbor_feat_list), dim=0)
