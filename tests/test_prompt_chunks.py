"""Unit tests for SD1.5 prompt chunking (no GPU required)."""
from types import SimpleNamespace

import torch

from generate import encode_prompt_chunks_sd15


class _FakeTok:
    bos_token_id = 1
    eos_token_id = 2
    model_max_length = 77

    def __call__(self, text, truncation=False, add_special_tokens=False):
        parts = [p for p in text.replace(",", " ").split() if p]
        return SimpleNamespace(input_ids=list(range(100, 100 + len(parts))))


class _FakeEnc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        b, s = input_ids.shape
        return (torch.randn(b, s, 8),)


def test_long_prompt_is_chunked_not_truncated():
    pipe = SimpleNamespace(tokenizer=_FakeTok(), text_encoder=_FakeEnc())
    long_prompt = ", ".join(["masterpiece"] * 120)
    p, n = encode_prompt_chunks_sd15(pipe, long_prompt, "low quality", device="cpu")
    assert p.shape[0] == 1
    assert n.shape[0] == 1
    assert p.shape[1] > 77
    assert p.shape[1] % 77 == 0
