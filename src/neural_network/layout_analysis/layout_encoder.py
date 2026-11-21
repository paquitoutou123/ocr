import torch
import torch.nn
import torchvision
import torchvision.models
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
        self.conv = torch.nn.Conv2d(2048, 256, 1)
        self.pos_encoding = sin_cos_positionnal_encoding.sinusoidal_positionnal_encoding(256, 400)
        self.attention = torch.nn.MultiheadAttention(256, 8)
        self.linear1 = torch.nn.Linear(256, 2048)
        self.linear2 = torch.nn.Linear(2048, 256)

    
    def aux1(self, input):
       
        array = input
        result = self.conv(array)
        result = torch.flatten(result)
        result = self.pos_encoding.sin_cos_positionnal_encoding(result)
        result = torch.stack(result)
        return result
    

    def res1(self, input):

        array = input
        result =  self.attention(array, array, array)
        result = result[0]
        return result


    def aux2(self, input):

        result = self.linear1(input)
        result = self.linear2(result)
        result = torch.nn.GELU()(result)
        return result


    def forward(self, input):

        array_list = []
        for k in range(input.size(0)):
            array = input[k]
            array = self.aux1(array)
            array = array + self.res1(array)
            array = self.aux2(array) + array
            array_list = array_list + [array]
        result = torch.stack(array_list)
        return result




class Ccff(torch.nn.Module):
    pass




class Efficient_hybrid_encoder(torch.nn.Module):
    """
    
    """

    def __init__(self, aifi, ccff):
        self.aifi = aifi
        self.ccff = ccff
        pass


    def forwardchannel1(self, input):
        pass


    def forwardchannel2(self, input):
        pass


    def forwardchannel3(self, input):
        pass


    def forward(self, input):

        pass



