# specially written to get direct helper classes for custom pretraining and testing
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embed: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    bias: bool = True
    learning_rate: float = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class SelfAttentionHead(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embed // config.n_head
        self.key = nn.Linear(config.n_embed, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embed, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embed, head_size, bias=config.bias)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embed // config.n_head
        self.heads = nn.ModuleList([SelfAttentionHead(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = MultiHeadSelfAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias),
            nn.Dropout(config.dropout)
        )
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class TextDataset:
    def __init__(self, text_path: str, train_split: float = 0.9):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, l: List[int]) -> str:
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str) -> tuple:
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

class Trainer:
    def __init__(self, model: GPT, dataset: TextDataset, config: GPTConfig):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    def train_step(self, batch_size: int) -> float:
        self.model.train()
        x, y = self.dataset.get_batch('train', batch_size, self.config.block_size, self.config.device)
        _, loss = self.model(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, batch_size: int, eval_iters: int) -> Dict[str, float]:
        self.model.eval()
        losses = {}
        for split in ['train', 'val']:
            losses[split] = 0
            for _ in range(eval_iters):
                x, y = self.dataset.get_batch(split, batch_size, self.config.block_size, self.config.device)
                _, loss = self.model(x, y)
                losses[split] += loss.item()
            losses[split] /= eval_iters
        return losses

    def train(self, num_steps: int, batch_size: int, eval_interval: int, eval_iters: int):
        for step in range(num_steps):
            loss = self.train_step(batch_size)
            
            if step % eval_interval == 0:
                losses = self.evaluate(batch_size, eval_iters)
                print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    def generate_text(self, prompt: str, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> str:
        self.model.eval()
        context = torch.tensor(self.dataset.encode(prompt), dtype=torch.long, device=self.config.device)
        context = context.unsqueeze(0)
        generated = self.model.generate(context, max_new_tokens, temperature, top_k)
        return self.dataset.decode(generated[0].tolist())