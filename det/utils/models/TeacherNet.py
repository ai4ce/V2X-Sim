from utils.models.base.NonIntermediateModelBase import NonIntermediateModelBase


class TeacherNet(NonIntermediateModelBase):
    def __init__(self, config):
        super(TeacherNet, self).__init__(config)

    def forward(self, bevs, maps=None, vis=None):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        # vis = vis.permute(0, 3, 1, 2)
        return self.stpn(bevs)
