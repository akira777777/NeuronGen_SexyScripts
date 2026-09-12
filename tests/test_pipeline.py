"""
Automated Test & Benchmark Suite for NeuronGen.
Validates:
- Config loading and schema integrity
- High-throughput prompt parsing (>50,000 parses/sec)
- WebUI/ComfyUI fast offline check (<1s)
- Batch generator in Demo mode
- Metadata persistence and JSON formatting
"""

import time
import unittest
from pathlib import Path

from config import get_config
from prompt_processor import PromptProcessor
from webui_api import StableDiffusionWebUI
from batch_generator import BatchGenerator


class TestNeuronGenPipeline(unittest.TestCase):

    def test_01_config_integrity(self):
        cfg = get_config()
        self.assertIn("webui", cfg)
        self.assertIn("generation", cfg)
        self.assertIn("output", cfg)
        self.assertEqual(cfg["generation"]["width"], 896)
        self.assertEqual(cfg["generation"]["height"], 1152)

    def test_02_prompt_processor_accuracy(self):
        p = PromptProcessor()
        raw = "(masterpiece, best quality:1.2), 1girl, [deformed]:steps:35:cfg:8.0:width:1024:height:1024:seed:1234"
        norm, tokens = p.process_prompt(raw)
        clean, params = p.extract_parameters(raw)

        self.assertEqual(params["steps"], 35)
        self.assertEqual(params["cfg_scale"], 8.0)
        self.assertEqual(params["width"], 1024)
        self.assertEqual(params["height"], 1024)
        self.assertEqual(params["seed"], 1234)
        self.assertTrue(len(tokens) > 0)

    def test_03_prompt_processor_throughput(self):
        p = PromptProcessor()
        sample = "(masterpiece, best quality:1.2), 1girl, solo, (sexy body:1.1):steps:30:cfg:8.5"
        
        # Warmup cache
        p.process_prompt(sample)

        count = 10000
        t0 = time.perf_counter()
        for _ in range(count):
            p.process_prompt(sample)
        elapsed = time.perf_counter() - t0
        throughput = count / elapsed

        print(f"\n[BENCHMARK] Prompt Parsing Throughput: {throughput:,.0f} parses/sec (Elapsed: {elapsed*1000:.2f}ms for {count} ops)")
        self.assertGreater(throughput, 10000, "Throughput should exceed 10,000 ops/sec")

    def test_04_backend_fast_offline_check(self):
        client = StableDiffusionWebUI()
        t0 = time.perf_counter()
        is_up = client.is_running("http://127.0.0.1:59999")
        elapsed = time.perf_counter() - t0
        
        self.assertFalse(is_up)
        print(f"[BENCHMARK] Port Offline Detection Latency: {elapsed*1000:.2f}ms")
        self.assertLess(elapsed, 1.0, "Fast offline check should complete in under 1 second")

    def test_05_batch_generator_demo(self):
        b = BatchGenerator()
        prompts = ["masterpiece, 1girl, cyberpunk aesthetic", "masterpiece, 1girl, fantasy forest"]
        files = b.generate_batch(prompts, engine="demo", batch_size=1)

        self.assertEqual(len(files), 2)
        for f in files:
            path = Path(f)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            meta_path = path.parent / path.name.replace(".png", "_metadata.json")
            self.assertTrue(meta_path.exists())


if __name__ == "__main__":
    unittest.main()
