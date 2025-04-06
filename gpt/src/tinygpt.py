import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

class SelfAttentionHead(nn.Module):
    """
    A single head of self-attention.

    Self Attention calculates and captures dependencies and relationships within
    the input sequences. allows model to identify and weigh importance of each 
    input sequence.

    How it works : 
    > split the input sequence into 3 vectors : 
        Q(Query), K(Key), V(Value)
    > calculate the dot product of Q and K, scaled by the square root of the
        dimensionality of Q.
    > apply softmax to the result of the dot product.

    xi -> token i of input sequence
    Wq, Wk, Wv -> learnable parameters/ projection matrices/ weight matrices
    q,k,v -> query, key, value sequences

    w(i,j) -> attention weight of token i relative to token j (either of i,j is constant, subject to token in consideration)
    w(i,j) = softmax(q(i) * k(j) / sqrt(d))

    i is constant in most cases, j being variable, as we compute attention score 
    for i-th token relative to all other tokens, including itself, capturing relation
    with all other tokens.

    where d is the dimensionality of the input sequence.
    """
    def __init__(self, n_embed, head_size):
        """
        args : 
            - n_embed : dimensionality of the input sequence.
            - head_size : dimensionality of the query, key, value sequences.
        returns : 
            - Nothing, but initializes the projection matrices for Q,K,V.
        """
        super().__init__()
        # linear transformations applied to input embeddings to produce q,k,v vectors
        self.key = nn.Linear(n_embed, head_size)
        self.query = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)

        # tril returns lower triangular part of matrix
        # creates a "tril" buffer
        self.register_buffer("tril", torch.tril(torch.ones(128,128)))

    def forward(self,x):
        """
        args : 
            - x : input sequence of shape (batch_size:128, seq_len, n_embed)
        returns :
            - out : weighted sum of value vectors using 
                    normalized attention weights.
        """
        # B: Batch Size, T: No. Of Tokens/ Sequence Length, C: Number of Embeddings
        B,T,C = x.shape 
        k = self.key(x)
        q = self.query(x) 
        v = self.value(x)

        # @ -> dot product
        wei = q @ k.transpose(-2,-1) * C ** -0.5

        # masking the attention weights (hide for future tokens)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        # Applies softmax along the last dimension (token dimension)
        wei = F.softmax(wei, dim=-1)

        # again, compute weighted sum of value vectors using 
        # normalized attention weights. (Final Output)
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """
    def __init__():
        pass 

    def forward():
        pass  

class TransformerBlock(nn.Module):
    """
    Uniting everything at a place
    """
    def __init__():
        pass 

    def forward():
        pass  

class GPT(nn.module):
    """
    The GPT model.
    """
    def __init__():
        pass 

    def forward():
        pass  


if __name__ == "__main__":
    print("Hello there!") 