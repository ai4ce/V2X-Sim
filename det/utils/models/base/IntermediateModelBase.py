from utils.models.backbone.Backbone import *
from utils.models.base.DetModelBase import DetModelBase


class IntermediateModelBase(DetModelBase):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True, num_agent=5):
        super().__init__(config, layer, in_channels, kd_flag, num_agent=num_agent)
        self.u_encoder = LidarEncoder(height_feat_size=in_channels)
        self.decoder = LidarDecoder(height_feat_size=in_channels)
