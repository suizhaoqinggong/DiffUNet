import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryExtractionModule(nn.Module):
    """
    PR25 Boundary Extraction Module (BEM).
    Multi-scale dilated conv -> GAP subtraction -> Sobel refinement.
    """

    def __init__(self, in_channels=320, out_channels=64, dilations=(1, 2, 4)):
        super().__init__()
        self.dilations = dilations
        self.dconvs = nn.ModuleList([
            nn.Conv3d(in_channels, in_channels, kernel_size=3,
                      padding=d, dilation=d, bias=False)
            for d in dilations
        ])

        # Fuse multi-scale boundary features
        self.fuse = nn.Sequential(
            nn.Conv3d(len(dilations) * in_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        # 3D Sobel-like kernels (one per spatial direction)
        # Shape: [1, 1, 3, 3, 3] -> applied with groups=out_channels
        self.register_buffer('sobel_x', self._make_sobel_kernel('x'))
        self.register_buffer('sobel_y', self._make_sobel_kernel('y'))
        self.register_buffer('sobel_z', self._make_sobel_kernel('z'))

    @staticmethod
    def _make_sobel_kernel(axis):
        """Build a 3x3x3 Sobel-like kernel for the given axis."""
        if axis == 'x':
            k = torch.tensor([
                [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                 [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                 [[1, 0, -1], [2, 0, -2], [1, 0, -1]]]
            ], dtype=torch.float32)
        elif axis == 'y':
            k = torch.tensor([
                [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                 [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                 [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]
            ], dtype=torch.float32)
        elif axis == 'z':
            k = torch.tensor([
                [[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]]
            ], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown axis {axis}")
        return k / k.abs().sum()

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        return: [B, out_channels, D, H, W]
        """
        S_list = []
        for dconv in self.dconvs:
            c = dconv(x)
            # Global Average Pooling -> subtract from feature
            g = F.adaptive_avg_pool3d(c, 1)  # [B, C, 1, 1, 1]
            s = c - g.expand_as(c)
            S_list.append(s)

        # Concatenate and fuse
        B_coarse = self.fuse(torch.cat(S_list, dim=1))

        # Sobel refinement (apply per-channel)
        B, C, D, H, W = B_coarse.shape
        sobel_x = self.sobel_x.repeat(C, 1, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(C, 1, 1, 1, 1)
        sobel_z = self.sobel_z.repeat(C, 1, 1, 1, 1)

        gx = F.conv3d(B_coarse, sobel_x, padding=1, groups=C)
        gy = F.conv3d(B_coarse, sobel_y, padding=1, groups=C)
        gz = F.conv3d(B_coarse, sobel_z, padding=1, groups=C)

        B_refined = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + 1e-8)
        return B_refined


class BoundaryDecoder(nn.Module):
    """
    Light-weight decoder that upsamples boundary features back to input resolution.
    Assumes 5 encoder downsampling stages (stride 2 each).
    """

    def __init__(self, in_channels=64, hidden_channels=64, num_upsample=5):
        super().__init__()
        blocks = []
        for _ in range(num_upsample):
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(hidden_channels),
                nn.LeakyReLU(inplace=True)
            ))
            in_channels = hidden_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class BoundarySupervisionHead(nn.Module):
    """BSM: 1x1 conv to produce per-class boundary logits."""

    def __init__(self, in_channels=64, num_classes=4):
        super().__init__()
        self.head = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(x)


class BoundaryGuidanceModule(nn.Module):
    """
    BGM: fuse boundary features into segmentation features.
    Implemented as an optional add-on for decoder stages.
    """

    def __init__(self, feat_channels, boundary_channels):
        super().__init__()
        self.conv_m = nn.Conv3d(feat_channels, feat_channels, kernel_size=1)
        self.conv_b = nn.Conv3d(boundary_channels, feat_channels, kernel_size=1)
        self.conv_out = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feat_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, feat, boundary_feat):
        """
        feat: [B, C, D, H, W]  decoder feature
        boundary_feat: [B, Cb, D, H, W]  boundary feature (same spatial size)
        """
        Fm = self.conv_m(feat)
        Fb = self.conv_b(boundary_feat)

        # Softmax attention over channels
        F_cat = Fm + Fb
        F_soft = F.softmax(F_cat, dim=1)
        F_sig = torch.sigmoid(F_cat)

        F_att = Fm * F_soft * F_sig
        F_out = F_att + Fm + Fb
        return self.conv_out(F_out)
