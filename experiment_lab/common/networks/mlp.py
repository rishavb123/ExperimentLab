"""A python module containing a basic MLP network class"""

from typing import List, Optional
from torch import nn


def create_mlp_network(
    layer_sizes: List[int],
    layer_activations: Optional[nn.Module | List[nn.Module]] = None,
    final_activation: Optional[nn.Module] = None,
) -> nn.Module:
    """Creates an mlp network using nn.Sequential.

    Args:
        layer_sizes (List[int]): The sizes of all the layers (including input and output layers).
        layer_activation_cls (Optional[nn.Module], optional): The activation function to use in between each row. Can be a single activation or a list. Defaults to None.
        final_activation_cls (Optional[nn.Module], optional): The final activation function to add for the last layer. Defaults to None.

    Returns:
        nn.Module: _description_
    """

    assert (
        len(layer_sizes) > 1
    ), "Length of layers sizes must be at least 2 to contain both the inputs and the outputs of the network."

    if isinstance(layer_activations, nn.Module):
        layer_activations = [layer_activations]

    mlp = nn.Sequential()
    for i in range(len(layer_sizes) - 2):
        mlp.append(
            nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i + 1])
        )
        if layer_activations is not None:
            if i >= len(layer_activations):
                mlp.append(layer_activations[-1])
            else:
                if layer_activations[i] is not None:
                    mlp.append(layer_activations[i])

    mlp.append(nn.Linear(in_features=layer_sizes[-2], out_features=layer_sizes[-1]))
    if final_activation is not None:
        mlp.append(final_activation)

    return mlp
