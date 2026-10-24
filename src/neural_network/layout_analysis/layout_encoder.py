import torch
import torch.nn
import torchvision

class Aifi(torch.nn):

    def __init__(self):
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