"""A python module containing a basic MLP network function"""

from typing import List, Type
from torch import nn


def create_mlp_network(
    layer_sizes: List[int],
    layer_activations: nn.Module | List[nn.Module | None] | None = None,
    final_activation: nn.Module | None = None,
    linear_layer_cls: Type[nn.Module] = nn.Linear,
) -> nn.Module:
    """Creates an mlp network using nn.Sequential.

    Args:
        layer_sizes (List[int]): The sizes of all the layers (including input and output layers).
        layer_activation_cls (nn.Module | None, optional): The activation function to use in between each row. Can be a single activation or a list. Defaults to None.
        final_activation_cls ([nn.Module] | None, optional): The final activation function to add for the last layer. Defaults to None.

    Returns:
        nn.Module: The mlp network.
    """

    assert (
        len(layer_sizes) > 1
    ), "Length of layers sizes must be at least 2 to contain both the inputs and the outputs of the network."

    if isinstance(layer_activations, nn.Module):
        layer_activations = [layer_activations]

    mlp = nn.Sequential()
    for i in range(len(layer_sizes) - 2):
        mlp.append(
            linear_layer_cls(in_features=layer_sizes[i], out_features=layer_sizes[i + 1])
        )
        if layer_activations is not None:
            activation = None
            
            if i >= len(layer_activations):
                activation = layer_activations[-1]
            else:
                if layer_activations[i] is not None:
                    activation = layer_activations[i]

            if activation is not None:
                mlp.append(activation)

    mlp.append(linear_layer_cls(in_features=layer_sizes[-2], out_features=layer_sizes[-1]))
    if final_activation is not None:
        mlp.append(final_activation)

    return mlp
