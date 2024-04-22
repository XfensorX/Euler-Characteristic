import torch
from torch import nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm = nn.LayerNorm(embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (seq_length, batch_size, embed_size)
        attn_output, _ = self.attention(x, x, x)
        x = self.relu(self.norm(attn_output + x))
        return x

class Transformer(nn.Module):
    def __init__(self, embed_size, output_size, img_x_size, img_y_size, heads):
        super().__init__()
        self.img_x_size = img_x_size
        self.img_y_size = img_y_size
        self.embed_size = embed_size

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(img_x_size * img_y_size, embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads)
        self.fc2 = nn.Linear(embed_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = x.view(-1, 1, self.embed_size)  # Reshape for the transformer block
        x = self.transformer_block(x)
        x = x.view(-1, self.embed_size)  # Flatten the output for the final linear layer

        x = self.fc2(x)
        return x