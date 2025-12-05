from torch import Tensor
from mmcv.cnn import Linear, build_activation_layer
from mmengine.model import BaseModule, ModuleList
from mmdet.utils import ConfigType
import torch

class MLP(BaseModule):
    """Very simple multi-layer perceptron (also called FFN) with relu. Mostly
    used in DETR series detectors.

    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers. As the last
            layer of MLP only contains FFN (Linear).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            act_cfg: ConfigType = dict(type='ReLU'),
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = ModuleList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLP.

        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def assert_no_nan_inf(x, name="tensor"):
    """Recursively check Tensor/list/tuple/dict for NaN and Inf separately."""

    # Case 1: Tensor
    if isinstance(x, torch.Tensor):
        has_nan = torch.isnan(x)
        has_inf = torch.isinf(x)

        if has_nan.any() or has_inf.any():
            msg_parts = [f"[ERROR] {name} contains invalid values"]

            if has_nan.any():
                msg_parts.append(f"- NaN at indices: {has_nan.nonzero().tolist()}")
            if has_inf.any():
                msg_parts.append(f"- Inf at indices: {has_inf.nonzero().tolist()}")

            raise ValueError("\n".join(msg_parts))

        return  # tensor OK

    # Case 2: list/tuple
    if isinstance(x, (list, tuple)):
        for i, item in enumerate(x):
            assert_no_nan_inf(item, f"{name}[{i}]")
        return

    # Case 3: dict
    if isinstance(x, dict):
        for k, v in x.items():
            assert_no_nan_inf(v, f"{name}.{k}")
        return

    # Anything else is unsupported
    raise TypeError(f"Unsupported type for {name}: {type(x)}")

