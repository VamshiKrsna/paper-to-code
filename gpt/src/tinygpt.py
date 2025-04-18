import math
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import torch

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

class MultiHeadSelfAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """
    def __init__(self, n_embed, n_head):
        """
        args:
            - n_embed : dimensionality of the input sequence.
            - n_head : number of heads of self-attention.
        returns:
            - Nothing, but initializes the heads of self-attention.
        """
        super().__init__()
        head_size = n_embed // n_head
        self.heads = nn.ModuleList(
            [SelfAttentionHead(n_embed, head_size = n_embed // n_head) for i in range(n_head)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        
    def forward(self, x):
        """
        args:
            - x : input sequence of shape (batch_size:128, seq_len, n_embed)
        returns:
            - self.proj(out) : weighted sum of value vectors using
                    normalized attention weights.
        """
        out = torch.cat(
            [
                h(x) for h in self.heads
            ], dim = -1
        )
        return self.proj(out)
          

class TransformerBlock(nn.Module):
    """
    Uniting everything at a place
    """
    def __init__(self, n_embed, n_head):
        """
        args: 
            - n_embed : dimensionality of the input sequence.
            - n_head : number of heads of self-attention.
        returns:
            Nothing.
        """
        super().__init__()
        self.sa = MultiHeadSelfAttention(
            n_embed, n_head 
        )
        self.ff = nn.Sequential(
            nn.Linear(
                n_embed, n_embed * 4
            ),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        """
        args:
            - x : input sequence of shape (batch_size:128, seq_len, n_embed)
        returns:
            - x : output sequence of shape (batch_size:128, seq_len, n_embed) 
                  after passing through the self-attention and feed-forward
                  layers.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x 

class GPT(nn.Module):
    """
    The GPT model.
    """
    def __init__(
        self, 
        vocab_size,
        block_size, 
        n_embed = 128,
        n_head = 4,
        n_layer = 2
    ):
        super().__init__() 
        self.token_embed = nn.Embedding(
            vocab_size,
            n_embed
        )
        self.pos_embed = nn.Embedding(
            block_size,
            n_embed
        )
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(n_embed, n_head) for i in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(
            n_embed
        )
        self.head = nn.Linear(
            n_embed,
            vocab_size
        )

    def forward(self,x):
        # B: Batch Size, T: No. Of Tokens/ Sequence Length, C: Number of Embeddings
        B, T = x.shape 
        token_emb = self.token_embed(x) # (B,T,C)
        pos = torch.arange(
            T, device = x.device
        )
        pos_emb = self.pos_embed(pos)[None, : , :] # (1,T,C)
        x = token_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.head(x) 
        return logits

class Helpers():
    def encode(self,s): 
        return [stoi[c] for c in s]

    def decode(self,l): 
        return ''.join([itos[i] for i in l])

    def get_batch(self,split, batch_size=64):
        data_ = train_data if split == "train" else val_data
        ix = torch.randint(len(data_) - block_size, (batch_size,))
        x = torch.stack([data_[i:i+block_size] for i in ix])
        y = torch.stack([data_[i+1:i+block_size+1] for i in ix])
        return x, y


if __name__ == "__main__":
    # Pretraining and testing out : 
    # Data prep : 
    with open("../data/sherlock_holmes.txt", "r", encoding= "cp1252") as f:
        text = f.read()
    print(f"Length of text: {len(text)} characters")

    # tokenization : 
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Number of unique characters: {vocab_size}")

    # char to int and vice versa : 
    stoi = {
        ch:i for i,ch in enumerate(chars)
    }
    itos = {
        i:ch for ch,i in stoi.items()
    }

    data = torch.tensor(Helpers().encode(text), dtype=torch.long)

    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    block_size = 128  # context length

    # Training : 
    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 1e-3
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device=device)

    def estimate_loss():
        model.eval()
        losses = {'train': 0, 'val': 0}
        for split in ['train', 'val']:
            x, y = Helpers().get_batch(split)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
            losses[split] += loss.item()
        model.train()
        return losses

    for step in range(10000):
        xb, yb = Helpers().get_batch("train")
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            losses = estimate_loss()
            print(f"Step {step} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")

    # Eval, Test : 
    perplexity = math.exp(losses["val"])
    print(f"Perplexity : {perplexity:.2f}")

    # Text Generation : 
    def generate(
        model, 
        start, 
        max_new_tokens = 30
    ):
        model.eval()
        context = torch.tensor(Helpers().encode(start), dtype=torch.long)[None, :].to(device)
        for i in range(max_new_tokens):
            x_cond = context[:,-block_size:]
            logits = model(x_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim = -1)
            next_id = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_id],dim = 1)
        return Helpers().decode(context[0].tolist())

    print(generate(model, "And brought miss hunter from "))
