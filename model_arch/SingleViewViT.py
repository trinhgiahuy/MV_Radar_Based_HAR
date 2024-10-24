import torch
import torch.nn as nn
import os

class SingleViewViT(nn.Module):
    def __init__(self, name, num_classes=7, patch_size=8, embed_dim=128, num_heads=8, num_encoder_layers=1, dtype=torch.float32):
        super(SingleViewViT, self).__init__()

        self.name = 'SingleViewViT_' + name
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.dtype = dtype

        # Max pooling layer to reduce input dimensionality
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Patch embedding for the radar input
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Input has 3 channels for radar data
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Further reduce size
            nn.Conv2d(16, embed_dim, kernel_size=3, stride=1, padding=1).to(dtype=dtype),
        )

        # Positional encodings
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, embed_dim, dtype=dtype))  # Fixed 512 patches from output

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dtype=dtype)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(embed_dim, 128, dtype=dtype)
        self.fc2 = nn.Linear(128, num_classes, dtype=dtype)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Print input shape for debugging
        print(f"Input shape: {x.shape}")

        # Ensure input is in the correct dtype and format (batch_size, 3, 32, 256)
        x = x.to(self.dtype)

        # Apply max pooling to reduce dimensionality
        x = self.maxpool(x)  # Output shape: (batch_size, 3, 16, 128)
        print(f"After max pooling: {x.shape}")

        # Patch embedding and flattening
        x = self.patch_embed(x)  # Apply patch embedding
        print(f"After patch embedding: {x.shape}")  # (batch_size, embed_dim, num_patches_h, num_patches_w)

        # Flatten and transpose the patches
        x = x.flatten(2).transpose(1, 2)  # Output shape: (batch_size, num_patches, embed_dim)
        print(f"After flattening and transposing patches: {x.shape}")

        # Dynamically calculate the number of patches
        num_patches_actual = x.size(1)
        print(f"Number of patches: {num_patches_actual}")

        # Update positional encodings to match the number of patches dynamically
        pos_embedding = self.pos_embedding[:, :num_patches_actual, :]

        # Add positional encodings
        x += pos_embedding

        # Pass through the transformer encoder
        x = self.transformer_encoder(x).mean(dim=1)  # Global average pooling over patches
        print(f"After transformer encoder: {x.shape}")

        # Pass through fully connected layers for classification
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        print(f"Output shape: {out.shape}")

        return out

    def get_name(self):
        return "SingleViewViT"

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path
