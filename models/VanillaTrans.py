import torch
from torch import nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),  # Changed to ReLU activation
            nn.Linear(embed_size * 4, embed_size)
        )
        self.relu = nn.ReLU()  # Using ReLU instead of GELU

    def forward(self, x):
        identity = x
        attn_output, _ = self.attention(x, x, x)
        x = attn_output + identity
        x = self.relu(self.norm1(x))
        
        identity = x
        x = self.feed_forward(x)
        x = x + identity
        x = self.relu(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, embed_size, output_size, img_x_size, img_y_size, heads, num_layers):
        super().__init__()
        self.embed_size = embed_size
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(embed_size, heads),
            *[TransformerBlock(embed_size, heads) for _ in range(num_layers-1)]
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(img_x_size * img_y_size, embed_size)
        self.fc2 = nn.Linear(embed_size, output_size)
        self.relu = nn.ReLU()  # Using ReLU instead of GELU

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.view(-1, 1, self.embed_size)
        x = self.transformer_blocks(x)
        x = x.view(-1, self.embed_size)
        x = self.fc2(x)
        return x
