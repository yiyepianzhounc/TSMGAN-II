import torch
import torch.nn as nn


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class Discriminator(nn.Module):
    def __init__(self, input_channel=2, ndf=16):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LayerNorm(100),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LayerNorm(50),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LayerNorm(25),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LayerNorm(12),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1)
        )

    def forward(self, x, y):
        x, y = x.permute(0, 1, 3, 2), y.permute(0, 1, 3, 2)
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)

