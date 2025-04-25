'''
Implementation of losses and evaluation metrics
'''

import torch
import torch.nn as nn


def chamfer_dist(x, y, weight_x=None, weight_y=None):
    dist = torch.cdist(x, y)
    if weight_x is None and weight_y is None:
        loss = dist.min(-2)[0].mean(-1) + dist.min(-1)[0].mean(-1)
    else:
        if weight_y is None:
            weight_y = weight_x
        loss = (dist.min(-2)[0] * weight_x).sum(-1) / weight_x.sum(-1) + \
               (dist.min(-1)[0] * weight_y).sum(-1) / weight_y.sum(-1)

    return loss


class ChamferLoss(nn.Module):
    """
    Chamfer Distance Loss
    Args:
        loss_weight (float, optional): Loss weight for Chamfer Distance. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(ChamferLoss, self).__init__()
        assert loss_weight >= 0, f'Invalid loss weight: {loss_weight}'
        self.loss_weight = loss_weight

    def forward(self, x, y, weight_x=None, weight_y=None):
        """
        Args:
            x (Tensor): of shape (N, V, C). point cloud x.
            y (Tensor): of shape (N, V, C). point cloud y.
            weight_x (Tensor): of shape (N, V, C). weight x.
            weight_y (Tensor): of shape (N, V, C). weight y.
        """
        return self.loss_weight * chamfer_dist(x, y, weight_x, weight_y).mean()
    
    
    
    
def hausdorff_dist(x, y, weight_x=None, weight_y=None):
    dist = torch.cdist(x, y)
    if weight_x is None and weight_y is None:
        loss = torch.max(dist.min(-2)[0].max(-1)[0], dist.min(-1)[0].max(-1)[0])
    else:
        if weight_y is None:
            weight_y = weight_x
        loss = torch.max(
            (dist.min(-2)[0] * weight_x).sum(-1) / weight_x.sum(-1),
            (dist.min(-1)[0] * weight_y).sum(-1) / weight_y.sum(-1)
        )

    return loss


class HausdorffLoss(nn.Module):
    """
    Hausdorff Distance Loss
    Args:
        loss_weight (float, optional): Loss weight for Hausdorff Distance. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(HausdorffLoss, self).__init__()
        assert loss_weight >= 0, f'Invalid loss weight: {loss_weight}'
        self.loss_weight = loss_weight

    def forward(self, x, y, weight_x=None, weight_y=None):
        """
        Args:
            x (Tensor): of shape (N, V, C). point cloud x.
            y (Tensor): of shape (N, V, C). point cloud y.
            weight_x (Tensor): of shape (N, V, C). weight x.
            weight_y (Tensor): of shape (N, V, C). weight y.
        """
        return self.loss_weight * hausdorff_dist(x, y, weight_x, weight_y).mean()