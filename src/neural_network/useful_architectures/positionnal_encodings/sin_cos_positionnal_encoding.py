import torch
import torchtext
import numpy


class sinusoidal_positionnal_encoding():

    def __init__(self, dimension, length):
        """
        it is meant to receive a length-uplet itself whose components are dimension-uplets
        """
        self.dimension = dimension
        self.length = length



    def sin_cos_positionnal_encoding(self, input):
        
        pos_encoding = [torch.zeros(self.dimension) for k in range(self.length)]
        for k in range(self.length):
            for j in range(self.dimension):
                if (j%2 == 0):
                    pos_encoding[k][j] = numpy.sin(k/10000**((2*j)/self.dimension))
                else:
                    pos_encoding[k][j] = numpy.cos(k/10000**((2*j)/self.dimension))

        return pos_encoding


