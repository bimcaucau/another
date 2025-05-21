# import copy
# from collections import OrderedDict

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from .HSFPN import HSFPN
#from ..core import register

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .utils import get_activation

__all__ = ['HybridHSFPNEncoder']

@register()
class HybridHSFPNEncoder(nn.Module):
    __share__ = ['eval_spatial_size']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 eval_spatial_size=None,
                 use_hsfpn=True):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim] * len(in_channels)
        self.out_strides = feat_strides

        # Project input channels to hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim))
            ])) for c in in_channels
        ])

        # Optional transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
        )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # Replace FPN + PAN with HS-FPN
        if use_hsfpn:
            self.hsfpn = HSFPN(
                in_channels_list=[hidden_dim] * len(in_channels),
                out_channels=hidden_dim,
                use_sdp=True,
                patch_size=(4, 4)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature
                )
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)

        # Step 1: Project backbone features
        proj_feats = [proj(f) for proj, f in zip(self.input_proj, feats)]

        # Step 2: Optional transformer encoder
        for idx, enc in zip(self.use_encoder_idx, self.encoder):
            B, C, H, W = proj_feats[idx].shape
            x = proj_feats[idx].flatten(2).permute(0, 2, 1)
            pos = getattr(self, f'pos_embed{idx}', None)
            if pos is not None:
                pos = pos.to(x.device)
            x = enc(x, pos_embed=pos)
            proj_feats[idx] = x.permute(0, 2, 1).reshape(B, C, H, W)

        # Step 3: HS-FPN fusion
        out_feats = self.hsfpn(proj_feats)

        return out_feats


# Supporting TransformerEncoder
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError("Unsupported activation")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, pos_embed=None):
        q = k = self.with_pos_embed(src, pos_embed)
        src2, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, src_mask=None, pos_embed=None):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, pos_embed=pos_embed)
        return src

print("[REGISTER] HybridHSFPNEncoder is being registered.")
