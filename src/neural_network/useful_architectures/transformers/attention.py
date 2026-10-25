import torch
import torch.nn
import numpy


class MultiHeadAttention():

    
    def scaled_dot_product_attention(Q, K, V):
        """
        Q: the Query
        K: the key
        V: the value
        """
        transposed = numpy.matrix.transpose(K)
        matmul = numpy.matmul(Q, transposed)

        d = (numpy.shape())[1]
        div = matmul/(numpy.sqrt(d))

        return (torch.nn.Softmax(div)*V)

    def projection():
        pass

    def concatenate():
        pass

    def multihead_attention():
        pass


