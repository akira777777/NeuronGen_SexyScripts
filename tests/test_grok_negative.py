"""Regression: Grok enhanced_negative must reach generation engines."""
from unittest.mock import MagicMock, patch

import generate


def test_execute_pipeline_uses_enhanced_negative(tmp_path):
    fake_client = MagicMock()
    fake_client.is_configured.return_value = True
    fake_client.enhance_prompt.return_value = (
        "beautiful woman selfie",
        "extra fingers, fused fingers, worst quality",
        "caption",
    )

    captured = {}

    def fake_diffusers(**kwargs):
        captured.update(kwargs)
        out = tmp_path / "out.png"
        out.write_bytes(b"x")
        return [str(out)]

    with patch.object(generate, "get_grok_client", return_value=fake_client), \
         patch.object(generate, "generate_image_diffusers", side_effect=fake_diffusers), \
         patch.object(generate, "PromptProcessor") as PP:
        proc = PP.return_value
        proc.process_prompt.return_value = ("beautiful woman selfie", {})
        proc.extract_parameters.return_value = ("", {})
        generate.execute_pipeline(
            prompt="girl selfie",
            engine="diffusers",
            output=str(tmp_path / "out.png"),
            negative_prompt="caller negative should be replaced",
        )

    assert "extra fingers" in captured.get("negative_prompt", "")
    fake_client.enhance_prompt.assert_called_once()
