import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register
from .HSFPN import HFP, SDP

class Conv(nn.Module):
    def __init__(self, cin, cout, k=1, s=1, p=None, g=1, act=nn.SiLU()):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2 if p is None else p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[Conv(c_, c_, 3, 1) for _ in range(n)])
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

@register()
class HSFPN_CSPPAN(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256, depth=1, use_sdp=True, mode='hsfpn'):
        super().__init__()
        self.mode = mode
        self.use_sdp = use_sdp and mode in ['hsfpn', 'both']

        # FPN: top-down
        self.fpn_stems = nn.ModuleList([Conv(c, out_channels, 1) for c in in_channels[::-1]])
        self.fpn_hfps = nn.ModuleList([HFP(out_channels) for _ in in_channels[::-1]]) if mode in ['hsfpn', 'both'] else None
        self.fpn_csps = nn.ModuleList([C3(out_channels * 2, out_channels, n=depth) for _ in in_channels[::-1][1:]])

        # PAN: bottom-up
        self.pan_stems = nn.ModuleList([Conv(out_channels, out_channels, 3, 2) for _ in in_channels[:-1]])
        self.pan_sdps = nn.ModuleList([SDP(out_channels, patch_size=(4, 4)) for _ in in_channels[1:]]) if self.use_sdp else None
        self.pan_csps = nn.ModuleList([C3(out_channels * 2, out_channels, n=depth) for _ in in_channels[1:]])

    def forward(self, feats):
        feats = feats[::-1]  # P5 -> P3
        fpn_feats = []

        for i, x in enumerate(feats):
            x = self.fpn_stems[i](x)
            if self.mode in ['hsfpn', 'both']:
                x = self.fpn_hfps[i](x)
            if i > 0:
                x = torch.cat([x, F.interpolate(fpn_feats[-1], size=x.shape[-2:], mode='nearest')], dim=1)
                x = self.fpn_csps[i - 1](x)
            fpn_feats.append(x)

        pan_feats = [fpn_feats[-1]]
        for i in range(len(fpn_feats) - 1):
            down = self.pan_stems[i](pan_feats[-1])
            if self.use_sdp:
                down = self.pan_sdps[i](down, fpn_feats[-2 - i])
            x = torch.cat([down, fpn_feats[-2 - i]], dim=1)
            x = self.pan_csps[i](x)
            pan_feats.append(x)

        return pan_feats[::-1]  # restore order P3 -> P5
