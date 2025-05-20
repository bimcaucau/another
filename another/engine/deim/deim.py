"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.projection_layer = nn.Conv2d(384, 256, kernel_size=1)
    def forward(self, x, targets=None):
        x = self.backbone(x)
        encoder_features = self.encoder(x)

        projected_features = []

        for feat in encoder_features:
            projected_features.append(self.projection_layer(feat))
        x = self.decoder(projected_features, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
