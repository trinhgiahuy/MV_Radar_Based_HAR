import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# SeparableConv2D block for lightweight convolutions
class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        nn.init.xavier_uniform_(self.depthwise.weight)
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.constant_(self.depthwise.bias, 0.2)
        nn.init.constant_(self.pointwise.bias, 0.2)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Residual Block for ResNet with optional attention mechanism
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, attention=False):
        super(ResNetBlock, self).__init__()
        self.downsample = downsample
        self.attention = attention

        self.conv1 = SeparableConv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SeparableConv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)

        if attention:
            self.attention_layer = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 8, 1),
                nn.ReLU(),
                nn.Conv2d(out_channels // 8, out_channels, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)

        out += identity
        out = F.relu(out)

        if self.attention:
            out = out * self.attention_layer(out)

        return out

# Single Stream model for single view radar data
class SingleStream(nn.Module):
    def __init__(self, num_classes=9):
        super(SingleStream, self).__init__()

        # Initial convolution layer
        self.conv1 = SeparableConv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks (including attention blocks for deeper layers)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, downsample=False)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, downsample=True)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, downsample=True, attention=True)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, downsample=True, attention=True)

        # Global Average Pooling and Final FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def _make_layer(self, in_channels, out_channels, num_blocks, downsample, attention=False):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride=2 if downsample else 1, downsample=downsample, attention=attention))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, attention=attention))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)
        return x

# Full Model for Single View Radar Data (Extending ResNet18Separable)
class ResNet18SeparableSingleViewAttention(nn.Module):
    def __init__(self, name='single_view_experiment', num_classes=9):
        super(ResNet18SeparableSingleViewAttention, self).__init__()
        self.name = "ResNet18SeparableSingleViewAttention" + name

        # Single stream model for single view radar data
        self.single_stream = SingleStream(num_classes=num_classes)

    def forward(self, x):
        # Pass through the single stream model
        return self.single_stream(x)
    
    def get_name(self):
        return "ResNet18SeparableSingleViewAttention" 
