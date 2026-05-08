import torch
import torch.nn.functional as F


def compute_boundary_gt(label, num_classes, kernel_size=3):
    """
    Generate binary boundary maps from segmentation labels using 3D erosion.

    Args:
        label: Long tensor of shape [B, H, W, D] or [B, 1, H, W, D]
        num_classes: number of classes (including background)
        kernel_size: erosion kernel size (default 3)

    Returns:
        boundary: Float tensor of shape [B, num_classes, H, W, D], values in {0, 1}
    """
    if label.dim() == 5:
        label = label.squeeze(1)

    B, H, W, D = label.shape
    device = label.device
    dtype = torch.float32

    boundaries = []
    for c in range(num_classes):
        mask = (label == c).to(dtype).unsqueeze(1)  # [B, 1, H, W, D]
        # 3D erosion: max_pool3d on negative values
        eroded = -F.max_pool3d(
            -mask,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        boundary = (mask - eroded).squeeze(1)  # [B, H, W, D]
        boundaries.append(boundary)

    return torch.stack(boundaries, dim=1)  # [B, num_classes, H, W, D]
