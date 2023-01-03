import torch.nn as nn
import torch

class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature_extractor = FeatureExtractor()
        self.head = Head(96, 10)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.flatten(start_dim=1)
        x = self.head(x)
        return x

class Inference(nn.Module):
    def __init__(self, classifier):
        super().__init__()

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        x = torch.argmax(x, dim=1)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(1, 12),
            ConvBlock(12, 12),
            nn.MaxPool2d(3),
            ConvBlock(12, 24),
            nn.MaxPool2d(3)
        )

    def forward(self, x):
        return self.block(x)

class Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_dim,50),
            nn.Linear(50, out_dim)
        )

    def forward(self, x):
        return self.block(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

