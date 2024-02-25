"""A python module for utilities functions."""

import collections.abc
from typing import Any, Dict


def default(val: Any, d: Any, null_val: Any = None) -> Any:
    """Defaults a value to something when it matches a null_val.

    Args:
        val (Any): The original value.
        d (Any): The value to set if val == null_val.
        null_val (Any, optional): The null value. Defaults to None.

    Returns:
        Any: The value defaulting to the chosen default.
    """
    return val if val != null_val else d


def deep_update(d: Dict[Any, Any], u: collections.abc.Mapping) -> Dict[Any, Any]:
    """Deep update for dictionaries.

    Args:
        d (Dict[Any, Any]): The initial dictionary.
        u (collections.abc.Mapping): The mapping to update the initial dictionary with.

    Returns:
        Dict[Any, Any]: The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d