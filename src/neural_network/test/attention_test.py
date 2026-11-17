import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3])) 

from src.neural_network.useful_architectures.transformers.attention import MultiHeadAttention, ScaledDotProductAttention


import torch
import torch.nn

#test to see if my implementation even works
def scaleddotproductattention_initial_test():

    Q = torch.randn(10, 512)
    K = torch.randn(10, 512)
    V = torch.randn(10, 512)

    
    return ScaledDotProductAttention(Q, V, K)
    




def scaleddotproductattention_equivalence_pytorch_test():

    Q = torch.randn(10, 512)
    K = torch.randn(10, 512)
    V = torch.randn(10, 512)
    
    mine_value = ScaledDotProductAttention(Q, K, V)
    pytorch = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    #bool = (torch.equal(pytorch, mine_value))
    diff = (mine_value - pytorch).abs()
    if (diff.max() < 10**(-6)):
        print("validated implementation\n")
    else:
        print("unvalidated implementation\n")

    

    

    

#test to see if my implementation matches pytorch's one 
def test_pytorch_equivalence():

    torch.manual_seed(0)

    # my implementation
    mine = MultiHeadAttention(total_dim=512, head_num=8)

    # pytorch's implementation
    ref = torch.nn.MultiheadAttention(512, 8, batch_first=True)

#    # weight copy
#    with torch.no_grad():
#       ref.in_proj_weight.copy_(
#            torch.cat([mine.W_q.weight, mine.W_k.weight, mine.W_v.weight], dim=0)
#        )
#       ref.out_proj.weight.copy_(mine.W_o.weight)

    
    ref.eval()

    #same entry for both
    x = torch.randn(3, 10, 512)

    out_mine = mine.forward(x[0], x[1], x[2])
    out_ref, _ = ref(x[0], x[1], x[2])

    # Comparison
    max_diff = (out_mine - out_ref).abs().max().item()
    success = torch.allclose(out_mine, out_ref, atol=1e-5)

    if success:
        print(f"SUCCÈS : same output (écart max = {max_diff:.2e})")
        return True
    else:
        print(f"ÉCHEC : écart max = {max_diff:.2e} (tolérance 1e-5)")
        print(f"   forme attendue : {tuple(out_ref.shape)}")
        print(f"   forme obtenue  : {tuple(out_mine.shape)}")
        return False



#print(scaleddotproductattention_initial_test())
#print(scaleddotproductattention_equivalence_pytorch_test())
test_pytorch_equivalence()