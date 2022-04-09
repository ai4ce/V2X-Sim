import torch

from utils.models.FusionBase import FusionBase


class MeanFusion(FusionBase):
    def __init__(self, n_channels, n_classes, num_agent=5):
        super().__init__(n_channels, n_classes, num_agent=num_agent)

    def fusion(self):
        return torch.mean(torch.stack(self.neighbor_feat_list), dim=0)
