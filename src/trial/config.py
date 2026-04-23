from .providers.base import BaseProvider

_default_provider: BaseProvider | None = None


def configure(provider: BaseProvider) -> None:
    global _default_provider
    _default_provider = provider


def get_provider() -> BaseProvider:
    if _default_provider is None:
        raise RuntimeError("No provider configured. Call trial.configure() first.")
    return _default_provider
