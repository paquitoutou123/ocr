import torch
import torch.nn
import torchvision

from useful_architectures.transformers.attention import MultiHeadAttention

class Aifi(torch.nn):
    """
    AIFI is just a restricted version of self-attention where Q=K=V
    """

    def __init__(self):
        pass

    def forward():
        pass

class Ccff(torch.nn):
    pass

class Efficient_hybrid_encoder(torch.nn):
    """
    
    """

    def __init__(self, aifi, ccff):
        self.aifi = aifi
        self.ccff = ccff
        pass