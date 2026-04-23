import os

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def complete(self, system: str, user: str) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is not installed. Install it with: pip install 'trial[openai]'"
            )

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content
