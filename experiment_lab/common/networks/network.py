"""A python module containing a basic MLP network function"""

from typing import Any, Callable, Dict, List, Type, TypeVar
import torch
from torch import nn

from experiment_lab.common.utils import default


def create_network(
    layer_cls: Type[nn.Module] | List[Type[nn.Module] | None] | None,
    n_layers: int,
    layer_kwargs: Dict[str, Any] | List[Dict[str, Any] | None] | None = None,
    constant_layer_kwargs: Dict[str, Any] | None = None,
    layer_activations: nn.Module | List[nn.Module | None] | None = None,
    final_activation: nn.Module | None = None,
    dropout_p: List[float | None] | float | None = None,
) -> nn.Module:

    network = nn.Sequential()

    constant_layer_kwargs_defaulted: Dict[str, Any] = default(constant_layer_kwargs, {})

    T = TypeVar("T")

    def get_from_lst(
        lst: T | List[T | None] | None, i: int, default_value: T | None = None
    ) -> T | None:
        val: T | None = None
        if lst is None:
            val = default_value
        elif type(lst) == list:
            if i < len(lst):
                val = lst[i]
            else:
                val = lst[-1]
        if val is None:
            return default_value
        else:
            return val

    for i in range(n_layers):
        layer_cls_i = get_from_lst(lst=layer_cls, i=i)
        layer_kwargs_i = get_from_lst(lst=layer_kwargs, i=i, default_value={})
        dropout_p_i = get_from_lst(lst=dropout_p, i=i)
        activation = (
            final_activation
            if i == n_layers - 1
            else get_from_lst(lst=layer_activations, i=i, default_value=None)
        )
        if layer_cls_i is not None and layer_kwargs_i is not None:
            network.append(
                layer_cls_i(**layer_kwargs_i, **constant_layer_kwargs_defaulted)
            )
        if activation is not None:
            network.append(activation)
        if dropout_p_i is not None and dropout_p_i > 0.0:
            network.append(nn.Dropout(p=dropout_p_i))

    return network


class ComplexNetwork(nn.Module):

    def __init__(
        self,
        module_lst: List[nn.Module],
        aggregator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        output_module: nn.Module,
    ) -> None:
        super().__init__()
        self.module_lst = module_lst
        self.aggregator = aggregator
        self.output_module = output_module

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        zs: List[torch.Tensor] = [m(x) for x, m in zip(xs, self.module_lst)]
        agg_z = zs[0]
        for z in zs[1:]:
            agg_z = self.aggregator(agg_z, z)
        return self.output_module(agg_z)


def create_complex_network(
    module_lst: List[nn.Module],
    aggregator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    output_module: nn.Module,
) -> ComplexNetwork:
    return ComplexNetwork(module_lst, aggregator, output_module)
