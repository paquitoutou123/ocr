import torch
import torch.nn
import torchvision
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from useful_architectures.transformers.attention import MultiHeadAttention
from  useful_architectures.positionnal_encodings import sin_cos_positionnal_encoding


class Aifi(torch.nn.Module):
    """
    AIFI is just a restricted version of self-attention where Q=K=V
    """

    def __init__(self):
        super(Aifi, self).__init__()
        self.conv = torch.nn.Conv2d(512, 256, 1)
        self.pos_encoding = sin_cos_positionnal_encoding.sinusoidal_positionnal_encoding(256, 400)
        

    def forward(self, input):
        for k in range(input.size(0)):
            
            array = input[k]
            result = self.conv(array)
            result = torch.flatten(result)
            result = self.pos_encoding.sin_cos_positionnal_encoding(result)
        
        return input

class Ccff(torch.nn.Module):
    pass

class Efficient_hybrid_encoder(torch.nn.Module):
    """
    
    """

    def __init__(self, aifi, ccff):
        self.aifi = aifi
        self.ccff = ccff
        pass



