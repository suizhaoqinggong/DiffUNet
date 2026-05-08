from .nnunet3d_denoise import get_nnunet3d_denoise
from .nnunet3d import get_nnunet3d
from .pr25_boundary import BoundaryExtractionModule, BoundaryDecoder, BoundarySupervisionHead
import torch
import torch.nn as nn

class DiffUNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 ddim_steps=3, rand_steps=1, bta=True,
                 use_pr25_boundary=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.edge_model = get_nnunet3d(in_chans=in_channels, out_chans=out_channels)
        self.denoise_model = get_nnunet3d_denoise(in_chans=in_channels, out_chans=out_channels,
                                          ddim_step=ddim_steps,
                                          rand_step=rand_steps,
                                          bta=bta)

        self.use_pr25_boundary = use_pr25_boundary
        if use_pr25_boundary:
            # BEM on embeddings[-1] (bottleneck): 320 channels after 6 stages
            self.bem = BoundaryExtractionModule(in_channels=320, out_channels=64)
            self.boundary_decoder = BoundaryDecoder(in_channels=64, hidden_channels=64, num_upsample=5)
            self.boundary_head = BoundarySupervisionHead(in_channels=64, num_classes=out_channels)
            # Spatial gate: learn per-pixel fusion ratio between pred and boundary
            self.boundary_gate = nn.Sequential(
                nn.Conv3d(out_channels * 2, out_channels, 1),
                nn.Sigmoid()
            )

    def forward(self, image, gt=None, ddim=False):
        pred_edge, embeddings = self.edge_model(image)

        pr25_boundary_pred = None
        if self.use_pr25_boundary:
            # Apply BEM on bottleneck (last embedding, 1/32 resolution)
            boundary_feat = self.bem(embeddings[-1])
            boundary_feat_up = self.boundary_decoder(boundary_feat)
            pr25_boundary_pred = self.boundary_head(boundary_feat_up)

        if ddim:
            pred = self.denoise_model(image, gt=gt,
                                        embeddings=embeddings,
                                        ddim=True)
            if self.use_pr25_boundary:
                gate = self.boundary_gate(torch.cat([pred, pr25_boundary_pred], dim=1))
                return gate * pred + (1 - gate) * pr25_boundary_pred
            else:
                return pred + pred_edge
        else:
            pred, uncertainty = self.denoise_model(image, gt=gt,
                                        embeddings=embeddings,
                                        ddim=False)
            if self.use_pr25_boundary:
                return pred, pr25_boundary_pred, uncertainty
            else:
                return pred, pred_edge, uncertainty
