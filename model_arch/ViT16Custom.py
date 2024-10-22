import torch
import torch.nn as nn
import os
from torchvision.models import vit_b_16, ViT_B_16_Weights
from einops.layers.torch import Rearrange

# Set the TORCH_HOME environment variable to a custom path where you have write permissions
os.environ['TORCH_HOME'] = '/mnt/slurm_nfs/h3trinh/.cache/torch'

class ViT16Custom(nn.Module):
    def __init__(self, name, num_classes=9, patch_size=(4, 16), embed_dim=768, num_heads=12, depth=12):
        super(ViT16Custom, self).__init__()

        self.name = name
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Input dimensions: (batch_size, 3, 32, 256) (3 channels, 32x256 heatmap)
        # Calculate the number of patches along the height and width
        self.num_patches_h = 32 // patch_size[0]
        self.num_patches_w = 256 // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Ensure patch size divides the input dimensions correctly
        assert (32 % patch_size[0] == 0) and (256 % patch_size[1] == 0), \
            "Patch size must divide input image dimensions perfectly."

        # Patch Embedding Layer
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
            nn.Linear(patch_size[0] * patch_size[1] * 3, embed_dim)  # Linear projection for each patch
        )

        # Positional Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification Head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Input x: (batch_size, 3, 32, 256)
        print(f"Input shape: {x.shape}")  # Debugging input shape
        
        # Patch Embedding: convert the input into patch embeddings
        x = self.patch_embed(x)  # Shape: (batch_size, num_patches, embed_dim)
        print(f"Patch embedding shape: {x.shape}")  # Debugging patch embedding shape

        # Add positional encoding
        x += self.pos_embedding

        # Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, num_patches, embed_dim)

        # Global Average Pooling over the patch dimension
        x = x.mean(dim=1)  # Shape: (batch_size, embed_dim)

        # Final classification layer
        x = self.fc(x)  # Shape: (batch_size, num_classes)
        
        return x

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path

    def get_name(self):
        return self.name