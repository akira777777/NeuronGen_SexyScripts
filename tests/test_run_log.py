"""Run log writes effective prompts for realism QA / future UI."""
from unittest.mock import MagicMock, patch
from pathlib import Path

import generate


def test_execute_pipeline_writes_run_log(tmp_path, monkeypatch):
    monkeypatch.setattr(generate, "RESULTS_DIR", tmp_path)
    fake_client = MagicMock()
    fake_client.is_configured.return_value = False

    def fake_diffusers(**kwargs):
        out = tmp_path / "out.png"
        out.write_bytes(b"x")
        return [str(out)]

    with patch.object(generate, "get_grok_client", return_value=fake_client), \
         patch.object(generate, "generate_image_diffusers", side_effect=fake_diffusers), \
         patch.object(generate, "PromptProcessor") as PP:
        proc = PP.return_value
        proc.process_prompt.return_value = ("pos prompt here", {})
        proc.extract_parameters.return_value = ("", {})
        generate.execute_pipeline(
            prompt="girl selfie",
            engine="diffusers",
            output=str(tmp_path / "out.png"),
            negative_prompt="plastic skin, extra fingers",
            cfg=6.0,
            seed=1001,
        )

    logs = list(tmp_path.glob("run_*.json"))
    assert len(logs) == 1
    data = __import__("json").loads(logs[0].read_text(encoding="utf-8"))
    assert data["run_id"]
    assert data["positive_prompt"] == "pos prompt here"
    assert "plastic skin" in data["negative_prompt"]
    assert data["cfg"] == 6.0
    assert data["seed"] == 1001
    assert data["engine"] == "diffusers"
