import torch
from torch import nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.projection = nn.Sequential(
            nn.Conv2d(1, embed_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)  # Flattening from the channel dimension
        )
        self.position_embeddings = nn.Parameter(torch.randn(1, self.n_patches, embed_size))

    def forward(self, x):
        x = self.projection(x)  # Converts (batch, 1, img_size, img_size) to (batch, embed_size, n_patches)
        x = x.permute(0, 2, 1)  # Reshapes to (batch, n_patches, embed_size)
        x += self.position_embeddings  # Broadcasting position embeddings across the batch
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm = nn.LayerNorm(embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.relu(self.norm(attn_output + x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_size, output_size, heads, num_layers):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, embed_size)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_size, heads) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, output_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_blocks(x)
        x = x.mean(dim=0)
        x = self.fc_out(x)
        return x