from dataclasses import dataclass, field
from typing import Protocol

# USD per million tokens: (input, output)
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICES.get(model, (0.0, 0.0))
    return input_tokens / 1e6 * inp + output_tokens / 1e6 * out


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LLM(Protocol):
    def complete(
        self, system: str, user: str, model: str, max_tokens: int = 4096
    ) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self) -> None:
        self._client = None

    def complete(
        self, system: str, user: str, model: str, max_tokens: int = 4096
    ) -> LLMResponse:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # SDK handles retries
        msg = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            cost_usd=cost_usd(model, msg.usage.input_tokens, msg.usage.output_tokens),
        )


@dataclass
class FakeLLM:
    responses: list[str]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(
        self, system: str, user: str, model: str, max_tokens: int = 4096
    ) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(self.responses.pop(0), 10, 10, 0.0)
