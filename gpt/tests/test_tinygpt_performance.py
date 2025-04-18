import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from src.tinygpt_jupyter import GPT, GPTConfig, TextDataset, Trainer

def plot_training_curves(train_losses, val_losses, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate(config, dataset_text, num_steps, batch_size, eval_interval):
    dataset = TextDataset(dataset_text, train_split=0.9)
    model = GPT(config).to(config.device)
    trainer = Trainer(model, dataset, config)
    
    train_losses = []
    val_losses = []
    
    for step in range(num_steps):
        loss = trainer.train_step(batch_size)
        train_losses.append(loss)
        
        if step % eval_interval == 0:
            losses = trainer.evaluate(batch_size, eval_iters=5)
            val_losses.append(losses['val'])
            print(f"Step {step}: train loss {loss:.4f}, val loss {losses['val']:.4f}")
    
    return train_losses, val_losses

def experiment_sequence_length():
    sequence_lengths = [32, 64, 128, 256]
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text_data = '\n'.join(dataset['text'])
    
    results = []
    for seq_len in sequence_lengths:
        config = GPTConfig(
            vocab_size=256,
            block_size=seq_len,
            n_embed=128,
            n_head=4,
            n_layer=4
        )
        train_losses, val_losses = train_and_evaluate(
            config, text_data, num_steps=1000, batch_size=32, eval_interval=100
        )
        results.append({
            'seq_len': seq_len,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
        
        plot_training_curves(
            train_losses,
            val_losses,
            f'Training Curves (Sequence Length = {seq_len})',
            f'sequence_length_{seq_len}.png'
        )

def experiment_model_size():
    model_sizes = [
        {'n_embed': 64, 'n_head': 2, 'n_layer': 2},
        {'n_embed': 128, 'n_head': 4, 'n_layer': 4},
        {'n_embed': 256, 'n_head': 8, 'n_layer': 6}
    ]
    
    dataset = load_dataset('squad', split='train')
    text_data = '\n'.join([q['question'] + ' ' + q['answers']['text'][0] for q in dataset])
    
    for size in model_sizes:
        config = GPTConfig(
            vocab_size=256,
            block_size=128,
            **size
        )
        train_losses, val_losses = train_and_evaluate(
            config, text_data, num_steps=1000, batch_size=32, eval_interval=100
        )
        
        plot_training_curves(
            train_losses,
            val_losses,
            f'Training Curves (Model Size: {size["n_embed"]})',
            f'model_size_{size["n_embed"]}.png'
        )

if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    
    print("Running sequence length experiments...")
    experiment_sequence_length()
    
    print("\nRunning model size experiments...")
    experiment_model_size()