"""Tests for list resolvers"""

from experiment_lab.common.resolvers.list_resolvers import dup, wrap_around, toggle


def test_dup():
    """Tests the dup resolver"""
    assert dup(3, "s") == ["s", "s", "s"]


def test_wrap_around():
    """Tests the wrap around resolver"""
    assert wrap_around(5, [1, 2, 3]) == [1, 2, 3, 1, 2]


def test_toggle():
    """Tests the toggle resolver"""
    assert toggle(10) == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
