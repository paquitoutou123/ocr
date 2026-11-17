import torch
import torch.nn
import numpy






def ScaledDotProductAttention(Q, K, V):
        """
        inputs:
            Q : the query matrix
            K: the key matrix
            V: the value matrix
        """

        transposed =torch.transpose(K, 0, 1)
        matmul = Q @ transposed

        d = (K.shape)[1]
        d = torch.tensor(d)
        div = matmul/(torch.sqrt(d))

        return ((torch.softmax(div, dim=1))@V)







class MultiHeadAttention(torch.nn.Module):

    def __init__(self, total_dim, head_num):
        """
        
        """        
        self.total_dim = total_dim
        self.head_num = head_num


    def project_into_heads(self, Q, K, V):
        """
        inputs:
            Q : the query matrix
            V : the value matrix
            K : the key matrix

        outputs:
            a tuple of lists of the projections of the Q V K matrices 
        """

        Q_heads = torch.chunk(Q, self.head_num, dim=1)
        V_heads = torch.chunk(V, self.head_num, dim=1)
        K_heads = torch.chunk(K, self.head_num, dim=1)

        return (Q_heads, K_heads, V_heads)
        

    def forward(self, Q, K, V):
        """
        inputs:
            Q : the query matrix
            V : the value matrix
            K : the key matrix
        """
        
        split_inputs = self.project_into_heads(Q, K, V)

        heads_list = []

        for k in range(self.head_num):
            result = ScaledDotProductAttention(split_inputs[0][k], split_inputs[1][k], split_inputs[2][k])
            heads_list.append(result)

        final_result =  torch.cat(heads_list, dim=1)

        return final_result
        

        





