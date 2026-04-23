from __future__ import annotations

from typing import Any, Callable

from .providers.base import BaseProvider

_default_provider: BaseProvider | None = None
_default_agent: Callable[[str], Any] | str | None = None


def configure(
    provider: BaseProvider,
    agent: Callable[[str], Any] | str | None = None,
) -> None:
    global _default_provider, _default_agent
    _default_provider = provider
    _default_agent = agent


def get_provider() -> BaseProvider:
    if _default_provider is None:
        raise RuntimeError("No provider configured. Call trial.configure() first.")
    return _default_provider


def get_agent() -> Callable[[str], Any] | str | None:
    return _default_agent
