import torch
import torch.nn
import numpy


class attention():


    def scaled_dot_product_attention(Q, K, V):
        """
        
        """
        
        transposed = numpy.matrix.transpose(K)

        matmul = numpy.matmul(Q, transposed)

        d = (numpy.shape())[1]

        div = matmul/(numpy.sqrt(d))

        return (torch.nn.Softmax(div)*V)





