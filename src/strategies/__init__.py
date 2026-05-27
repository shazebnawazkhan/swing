"""
strategies/__init__.py
----------------------
Strategy registry.  Use the ``@register_strategy`` decorator on any Strategy
subclass to make it discoverable.  Importing this package auto-registers all
built-in strategies via the module imports at the bottom.

Usage
-----
    from strategies import all_strategies, get_strategy, list_strategy_names

    # Iterate over every registered strategy
    for StrategyCls in all_strategies():
        strat = StrategyCls()
        ...

    # Get one strategy by name
    cls = get_strategy("Delivery + OI")
    strat = cls()
"""

from .base import BacktestResult, Strategy, StrategyParams, TradeRecord

_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    """
    Class decorator.  Registers a Strategy subclass under its params.name key.

        @register_strategy
        class MyStrategy(Strategy):
            params = MyParams(name="My Strategy")
            ...
    """
    key = cls.params.name
    if key in _REGISTRY:
        raise ValueError(
            f"A strategy named {key!r} is already registered. "
            "Each strategy must have a unique name."
        )
    _REGISTRY[key] = cls
    return cls


def get_strategy(name: str) -> type[Strategy]:
    """Return the Strategy class registered under ``name``."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Strategy {name!r} not found. "
            f"Available: {list_strategy_names()}"
        )
    return _REGISTRY[name]


def all_strategies() -> list[type[Strategy]]:
    """Return all registered Strategy classes."""
    return list(_REGISTRY.values())


def list_strategy_names() -> list[str]:
    """Return names of all registered strategies."""
    return list(_REGISTRY.keys())


# ── Auto-register built-in strategies ─────────────────────────────────────────
# Import order controls display order in comparison tables.
from . import delivery_oi  # noqa: E402, F401
from . import volume_ema   # noqa: E402, F401
from . import ema_bb       # noqa: E402, F401


__all__ = [
    "Strategy",
    "StrategyParams",
    "BacktestResult",
    "TradeRecord",
    "register_strategy",
    "get_strategy",
    "all_strategies",
    "list_strategy_names",
]
