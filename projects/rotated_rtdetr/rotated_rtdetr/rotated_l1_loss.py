import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
# 注意：你的文件里如果相对路径不对，请替换为 from mmdet.models.losses.utils import weighted_loss
# from .utils import weighted_loss
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def rotated_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Rotated L1 loss with periodic angle handling.

    Args:
        pred (Tensor): The prediction, typically shape (..., 5) for (cx, cy, w, h, angle).
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    
    # ---------------- 改进部分开始 ----------------
    # 1. 将预测值和目标值切分为位置参数 (前4维) 和 角度参数 (第5维)
    # 使用切片 [..., 4:] 保持维度，以便后续 cat 拼接
    pred_pos, pred_angle = pred[..., :4], pred[..., 4:]
    target_pos, target_angle = target[..., :4], target[..., 4:]

    # 2. 对前 4 维 (cx, cy, w, h) 进行普通的 L1 计算
    loss_pos = torch.abs(pred_pos - target_pos)

    # 3. 对第 5 维 (angle) 进行周期性最短路径计算
    angle_diff = pred_angle - target_angle
    # 使用公式: (Δθ + π/2) mod π - π/2
    # 这里使用 torch.remainder 保证对负数的取模行为在数学上是正确的
    angle_diff_periodic = torch.remainder(
        angle_diff + math.pi / 2, 
        math.pi
    ) - math.pi / 2
    loss_angle = torch.abs(angle_diff_periodic)

    # 4. 将位置 loss 和角度 loss 沿最后一个维度拼接回原始形状 (..., 5)
    loss = torch.cat([loss_pos, loss_angle], dim=-1)
    # ---------------- 改进部分结束 ----------------

    return loss


@MODELS.register_module()
class RotatedL1Loss(nn.Module):
    """Rotated L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
            
        # ---------------- 改进部分开始 ----------------
        # 调用新写的 rotated_l1_loss 函数，替换原来的 l1_loss
        loss_bbox = self.loss_weight * rotated_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        # ---------------- 改进部分结束 ----------------
        
        return loss_bbox