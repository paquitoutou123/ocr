import torch
import torch.nn
import torchvision

from neural_network.layout_analysis.backbone import Backbone

class Layout_analysis():
    """
    the selected architecture is RT-DETR

    backbone : this is initially a cnn, but can also be vision transformer...
    layout_encoder : a vision transformer encoder
    layout_decoder : a vision transformer decoder
    """

    def __init__(self, backbone, layout_encoder, layout_decoder):
        self.backbone = backbone
        self.layout_encoder = layout_encoder
        self.layout_decoder = layout_decoder
        pass

    def forward(self):
        
        pass


