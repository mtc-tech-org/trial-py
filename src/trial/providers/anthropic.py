import os

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def complete(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic is not installed. Install it with: pip install 'trial[anthropic]'"
            )

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text
