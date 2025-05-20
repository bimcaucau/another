import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import copy  
from collections import OrderedDict  
from ..core import register  
from .HSFPN import HFP, SDP
from .common import get_activation
from .hybrid_encoder import TransformerEncoder, TransformerEncoderLayer
from ..backbone.csp_darknet import autopad




class Conv(nn.Module):
    def __init__(self, cin, cout, k=1, s=1, p=None, g=1, act='silu') -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act='silu'):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='silu'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1, act=act)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

@register()  
class HybridHSFPNEncoder(nn.Module):  
    __share__ = ['eval_spatial_size', ]  
      
    def __init__(self,  
                 in_channels=[256, 512, 1024],  
                 feat_strides=[8, 16, 32],  
                 hidden_dim=384,  
                 nhead=8,  
                 dim_feedforward=2048,  
                 dropout=0.0,  
                 enc_act='gelu',  
                 use_encoder_idx=[2],  
                 num_encoder_layers=1,  
                 pe_temperature=10000,  
                 expansion=1.0,  
                 depth_mult=1.0,  
                 act='silu',  
                 eval_spatial_size=None,  
                 version='dfine',  
                 use_hfp=True,  
                 use_sdp=True,  
                 csp_depth=1,  
                 mode='hsfpn'  
                 ):  
        super().__init__()  
        self.in_channels = in_channels 
        print(f"[DEBUG] Encoder initialized with in_channels: {self.in_channels}")
        self.feat_strides = feat_strides  
        self.hidden_dim = hidden_dim  
        self.use_encoder_idx = use_encoder_idx  
        self.num_encoder_layers = num_encoder_layers  
        self.pe_temperature = pe_temperature  
        self.eval_spatial_size = eval_spatial_size  
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]  
        self.out_strides = feat_strides  
        self.use_hfp = use_hfp and mode in ['hsfpn', 'both']  
        self.use_sdp = use_sdp and mode in ['hsfpn', 'both']  
        self.mode = mode  
          
        # Channel projection (from HybridEncoder)  
        self.input_proj = nn.ModuleList()  
        for in_channel in in_channels:  
            proj = nn.Sequential(OrderedDict([  
                ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),  
                ('norm', nn.BatchNorm2d(hidden_dim))  
            ]))  
            self.input_proj.append(proj)  
          
        # Transformer encoder (from HybridEncoder)  
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
          
        # HFP modules (from HSFPN_CSPPAN)  
        if self.use_hfp:  
            self.hfp_modules = nn.ModuleList([  
                HFP(hidden_dim) for _ in in_channels  
            ])  
          
        # Top-down FPN with C3 blocks (from HSFPN_CSPPAN)  
        self.lateral_convs = nn.ModuleList()  
        self.fpn_blocks = nn.ModuleList()  
        for _ in range(len(in_channels) - 1, 0, -1):  
            self.lateral_convs.append(Conv(hidden_dim, hidden_dim, 1, 1))  
            self.fpn_blocks.append(C3(hidden_dim * 2, hidden_dim, n=csp_depth))  
          
        # Bottom-up PAN with SDP and C3 blocks (from HSFPN_CSPPAN)  
        self.downsample_convs = nn.ModuleList()  
        self.pan_blocks = nn.ModuleList()  
        if self.use_sdp:  
            self.pan_sdps = nn.ModuleList()  
          
        for _ in range(len(in_channels) - 1):  
            self.downsample_convs.append(Conv(hidden_dim, hidden_dim, 3, 2))  
            if self.use_sdp:  
                self.pan_sdps.append(SDP(hidden_dim, patch_size=(4, 4)))  
            self.pan_blocks.append(C3(hidden_dim * 2, hidden_dim, n=csp_depth))  
          
        self._reset_parameters()  
      
    def _reset_parameters(self):  
        # Initialize weights  
        for m in self.modules():  
            if isinstance(m, nn.Conv2d):  
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
      
    def forward(self, feats):  
        # Process input features  
        proj_feats = []  
        for i, feat in enumerate(feats):  
            print(f"[DEBUG] Feature {i} shape: {feat.shape}")  
            print(f"[DEBUG] Input proj {i} weight shape: {self.input_proj[i][0].weight.shape}")
            x = self.input_proj[i](feat)  
            proj_feats.append(x)  
          
        # Apply transformer encoder to selected feature levels  
        for i, idx in enumerate(self.use_encoder_idx):  
            if idx < len(proj_feats):  
                # Prepare for transformer  
                h, w = proj_feats[idx].shape[-2:]  
                pos_embed = self.build_2d_sincos_position_embedding(  
                    w, h, self.hidden_dim, self.pe_temperature).to(proj_feats[idx].device)  
                  
                # Reshape for transformer  
                src = proj_feats[idx].flatten(2).permute(2, 0, 1)  # HW, B, C  
                pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # HW, B, C  
                  
                # Apply transformer  
                memory = self.encoder[i](src, pos_embed=pos_embed)  
                  
                # Reshape back  
                memory = memory.permute(1, 2, 0).reshape(proj_feats[idx].shape)  
                proj_feats[idx] = memory  
          
        # Apply HFP modules if enabled  
        if self.use_hfp:  
            for i in range(len(proj_feats)):  
                proj_feats[i] = self.hfp_modules[i](proj_feats[i])  
          
        # Top-down path (FPN)  
        fpn_feats = [proj_feats[-1]]  # Start with the deepest feature  
        for i in range(len(proj_feats) - 1):  
            # Get features from the previous level  
            top_feat = fpn_feats[-1]  
            # Get features from the current level  
            lateral_feat = proj_feats[-(i+2)]  
            # Upsample the top features  
            top_feat_upsampled = F.interpolate(  
                top_feat, size=lateral_feat.shape[-2:], mode='nearest')  
            # Concatenate and process  
            x = torch.cat([lateral_feat, top_feat_upsampled], dim=1)  
            x = self.fpn_blocks[i](x)  
            fpn_feats.append(x)  
          
        # Bottom-up path (PAN)  
        pan_feats = [fpn_feats[-1]]  # Start with the smallest feature  
        for i in range(len(fpn_feats) - 1):  
            # Downsample the previous feature  
            down = self.downsample_convs[i](pan_feats[-1])  
            # Apply SDP if enabled  
            if self.use_sdp:  
                down = self.pan_sdps[i](down, fpn_feats[-(i+2)])  
            # Concatenate and process  
            x = torch.cat([down, fpn_feats[-(i+2)]], dim=1)  
            x = self.pan_blocks[i](x)  
            pan_feats.append(x)  
          
        # Return features in the original order (P3 -> P5)  
        return pan_feats[::-1]  
      
    @staticmethod  
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):  
        """  
        Build 2D sine-cosine position embedding.  
        """  
        grid_w = torch.arange(int(w), dtype=torch.float32)  
        grid_h = torch.arange(int(h), dtype=torch.float32)  
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')  
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'  
        pos_dim = embed_dim // 4  
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim  
        omega = 1. / (temperature ** omega)  
  
        out_w = grid_w.flatten()[..., None] @ omega[None]  
        out_h = grid_h.flatten()[..., None] @ omega[None]  
  
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]