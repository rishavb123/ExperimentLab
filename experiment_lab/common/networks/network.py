"""A python module containing functions to create generic networks."""

import collections.abc
from typing import Any, Callable, Dict, Sequence, Type, TypeVar
from typing_extensions import get_origin
import torch
from torch import nn

from experiment_lab.common.utils import default


def create_network(
    layer_cls: Type[nn.Module] | Sequence[Type[nn.Module] | None] | None,
    n_layers: int,
    layer_kwargs: Dict[str, Any] | Sequence[Dict[str, Any] | None] | None = None,
    constant_layer_kwargs: Dict[str, Any] | None = None,
    layer_activations: nn.Module | Sequence[nn.Module | None] | None = None,
    final_activation: nn.Module | None = None,
    dropout_p: Sequence[float | None] | float | None = None,
) -> nn.Module:
    """A generic create sequential network function

    Args:
        layer_cls (Type[nn.Module] | Sequence[Type[nn.Module]  |  None] | None): The layer class type(s) to use for each layer in the network.
        n_layers (int): The number of layers.
        layer_kwargs (Dict[str, Any] | Sequence[Dict[str, Any]  |  None] | None, optional): The kwargs to pass to the each layer. Defaults to None.
        constant_layer_kwargs (Dict[str, Any] | None, optional): The kwargs to pass to all the layers. Defaults to None.
        layer_activations (nn.Module | Sequence[nn.Module  |  None] | None, optional): The activation function to use after each layer. Defaults to None.
        final_activation (nn.Module | None, optional): The activation function to use at the end. Defaults to None.
        dropout_p (Sequence[float | None] | float | None, optional): The probability of dropout for each node. Defaults to None.

    Returns:
        nn.Module: The full network torch module.
    """

    network = nn.Sequential()

    constant_layer_kwargs_defaulted: Dict[str, Any] = default(constant_layer_kwargs, {})

    T = TypeVar("T")

    def get_from_lst(
        lst: T | Sequence[T | None] | None, i: int, default_value: T | None = None
    ) -> T | None:
        val: T | None = None
        if lst is None:
            val = default_value
        elif isinstance(lst, collections.abc.Sequence):
            if i < len(lst):
                val = lst[i]
            else:
                val = lst[-1]
        else:
            val = lst  # type: ignore
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
        if i < n_layers - 1 and dropout_p_i is not None and dropout_p_i > 0.0:
            network.append(nn.Dropout(p=dropout_p_i))

    return network


class ComplexNetwork(nn.Module):
    """A class for complex multi input networks."""

    def __init__(
        self,
        module_lst: Sequence[nn.Module],
        aggregator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        output_module: nn.Module,
    ) -> None:
        """The constructor for the multi input network.

        Args:
            module_lst (Sequence[nn.Module]): The list of modules to apply before aggregation.
            aggregator (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The aggregator to use on the tensors.
            output_module (nn.Module): The module to apply after aggregation.
        """
        super().__init__()
        self.module_lst = module_lst
        self.aggregator = aggregator
        self.output_module = output_module

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        """The forward function of the complex network.

        Args:
            xs (Sequence[torch.Tensor]): The inputs to the network.

        Returns:
            torch.Tensor: The output of the network.
        """
        zs: Sequence[torch.Tensor] = [m(x) for x, m in zip(xs, self.module_lst)]
        agg_z = zs[0]
        for z in zs[1:]:
            agg_z = self.aggregator(agg_z, z)
        return self.output_module(agg_z)


def create_complex_network(
    module_lst: Sequence[nn.Module],
    aggregator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    output_module: nn.Module,
) -> ComplexNetwork:
    """Creates an instance of the complex multi input network.

        Args:
            module_lst (Sequence[nn.Module]): The list of modules to apply before aggregation.
            aggregator (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The aggregator to use on the tensors.
            output_module (nn.Module): The module to apply after aggregation.

    Returns:
        ComplexNetwork: The complex multi input network.
    """
    return ComplexNetwork(module_lst, aggregator, output_module)
