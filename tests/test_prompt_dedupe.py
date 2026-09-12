# -*- coding: utf-8 -*-
"""Tests for prompt token deduplication."""
from prompt_processor import dedupe_prompt_tokens, _fast_process_prompt, PromptProcessor


def test_dedupe_plain_quality_tags():
    src = "masterpiece, best quality, foo, masterpiece, best quality"
    out = dedupe_prompt_tokens(src)
    parts = [p.strip() for p in out.split(",")]
    assert parts.count("masterpiece") == 1
    assert parts.count("best quality") == 1
    assert "foo" in parts
    assert parts.index("masterpiece") < parts.index("best quality") < parts.index("foo")


def test_dedupe_casefold_same_keyword():
    out = dedupe_prompt_tokens("masterpiece, Masterpiece, MASTERPIECE")
    parts = [p.strip() for p in out.split(",") if p.strip()]
    assert len(parts) == 1
    assert parts[0].casefold() == "masterpiece"


def test_dedupe_higher_weight_wins():
    norm, tokens = _fast_process_prompt("masterpiece, (masterpiece:1.2), foo")
    assert "(masterpiece:1.2)" in norm
    kws = [t[1].casefold() for t in tokens]
    assert kws.count("masterpiece") == 1
    mp = [t for t in tokens if t[1].casefold() == "masterpiece"][0]
    assert abs(mp[2] - 1.2) < 1e-6
    assert "foo" in norm


def test_dedupe_weighted_beats_plain_reverse_order():
    out = dedupe_prompt_tokens("(masterpiece:1.2), masterpiece")
    assert out.strip() == "(masterpiece:1.2)"


def test_process_prompt_uses_dedupe():
    pp = PromptProcessor()
    norm, toks = pp.process_prompt("best quality, foo, best quality")
    assert sum(1 for t in toks if t["keyword"].casefold() == "best quality") == 1
    assert "foo" in norm
