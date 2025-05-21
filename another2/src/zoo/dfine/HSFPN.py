import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core import register
import math

@register()
class HFP(nn.Module):
    def __init__(self, channels, alpha=0.25, pool_size=16):
        super().__init__()
        self.alpha = alpha
        self.pool_size = pool_size
        self.conv_spatial = nn.Conv2d(channels, 1, kernel_size=1)
        self.conv_channel_gap = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        self.conv_channel_gmp = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        self.conv_channel_final = nn.Conv2d(2 * channels, channels, kernel_size=1, groups=channels)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def high_pass_filter(self, x):
        B, C, H, W = x.shape
        H_pad = 2**math.ceil(math.log2(H))  
        W_pad = 2**math.ceil(math.log2(W))  

        padded_x = F.pad(x, (0, W_pad - W, 0, H_pad - H))  


        freq = torch.fft.fft2(padded_x, norm='ortho')
        mask = torch.ones_like(freq)
        h_cutoff = int(H * self.alpha)
        w_cutoff = int(W * self.alpha)
        mask[..., :h_cutoff, :w_cutoff] = 0
        filtered = freq * mask
        hf = torch.fft.ifft2(filtered, norm='ortho').real

        if H != H_pad or W != W_pad:  
            hf = hf[:, :, :H, :W]  
            
        return hf
        #return hf
    
    def forward(self, x):
        hf = self.high_pass_filter(x)
        gap = F.adaptive_avg_pool2d(hf, (self.pool_size, self.pool_size))
        gmp = F.adaptive_max_pool2d(hf, (self.pool_size, self.pool_size))
        gap_weight = self.conv_channel_gap(F.relu(gap)).mean(dim=[2, 3], keepdim=True)
        gmp_weight = self.conv_channel_gmp(F.relu(gmp)).mean(dim=[2, 3], keepdim=True)
        channel_weight = self.conv_channel_final(torch.cat([gap_weight, gmp_weight], dim=1))
        channel_weight = torch.sigmoid(channel_weight)
        spatial_mask = torch.sigmoid(self.conv_spatial(hf))
        if spatial_mask.shape != x.shape:  
            spatial_mask = F.interpolate(spatial_mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)  
        
        # Apply attention  
        out = x * channel_weight + x * spatial_mask  
        out = self.conv_out(out)  
        return out
        # out = x * channel_weight + x * spatial_mask
        # out = self.conv_out(out)
        # return out

@register()
class SDP(nn.Module):
    def __init__(self, channels, patch_size):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.patch_size = patch_size

    def forward(self, lower_feat, upper_feat):
        B, C, H, W = lower_feat.size()
        upper_feat = F.interpolate(upper_feat, size=(H, W), mode='nearest')

        Q = self.query_conv(lower_feat)
        K = self.key_conv(upper_feat)
        V = self.value_conv(upper_feat)

        # Divide into patches
        patch_H, patch_W = self.patch_size
        unfold_Q = F.unfold(Q, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        unfold_K = F.unfold(K, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        unfold_V = F.unfold(V, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        # Compute attention
        attn_scores = torch.matmul(unfold_Q, unfold_K.transpose(-1, -2)) / (C ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_probs, unfold_V)

        # Fold back to spatial map
        attended = attended.transpose(1, 2)
        attended = F.fold(attended, output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return lower_feat + attended

@register()
class HSFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, use_sdp=True, patch_size=(4, 4)):
        super().__init__()
        self.use_sdp = use_sdp
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        self.hfp_modules = nn.ModuleList([
            HFP(out_channels) for _ in in_channels_list
        ])
        if self.use_sdp:
            self.sdp_modules = nn.ModuleList([
                SDP(out_channels, patch_size=patch_size) if i < len(in_channels_list) - 1 else None
                for i in range(len(in_channels_list))
            ])

    def forward(self, inputs):
        laterals = [l_conv(x) for l_conv, x in zip(self.lateral_convs, inputs)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')
            laterals[i - 1] += upsampled

        for i in range(len(laterals)):
            laterals[i] = self.hfp_modules[i](laterals[i])

        if self.use_sdp:
            for i in range(len(laterals) - 1):
                laterals[i] = self.sdp_modules[i](laterals[i], laterals[i + 1])

        outs = [self.output_convs[i](feat) for i, feat in enumerate(laterals)]
        return outs
