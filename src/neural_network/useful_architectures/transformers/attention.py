import torch
import torch.nn
import numpy



class ScaledDotProductAttention():

    def __init__(self, Q, K, V):
        """
        Q: the Query
        K: the key
        V: the value
        """
        self.Q = Q
        self.K = K
        self.V = V



    def forward(self):

        transposed = numpy.matrix.transpose(self.K)
        matmul = numpy.matmul(self.Q, transposed)

        d = (numpy.shape())[1]
        div = matmul/(numpy.sqrt(d))

        return (torch.nn.Softmax(div)*(self.V))







class MultiHeadAttention():

    def __init__(self, total_dim, total_Q, total_V, total_K, head_num):
        """
        
        """        
        self.total_dim = total_dim
        self.total_Q = total_Q
        self.total_V = total_V
        self.total_K = total_K
        self.head_num = head_num


    def project_into_heads(self):

        W_heads = torch.chunk(self.Q.weight, self.head_num, dim=1)
        Q_heads = torch.chunk(self.V.weight, self.head_num, dim=1)
        K_heads = torch.chunk(self.K.weight, self.head_num, dim=1)

        return (W_heads, Q_heads, K_heads)
        

    def forward(self):
        
        heads_list = []
        for k in range(self.head_num):
            single_head_attention = ScaledDotProductAttention(self.project_into_heads()[0], self.project_into_heads()[1], self.project_into_heads()[2])
            result = single_head_attention.forward()
            heads_list.append(result)

        final_result =  torch.cat(heads_list, dim=1)

        return final_result
        

        





