#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
test_providers.py — offline tests for provider-side param translation.

Runs with pure stdlib (no network, no openai/boto3 required): exercises the
pure functions in providers.py that decide which OpenAI-style params survive
for a given model and how they're reshaped per API surface.

Covers the frontier-reasoning-model gap: temperature/top_p are dropped for
GPT-5.x/6 models, while their replacements — reasoning_effort and verbosity —
pass through untouched on Chat Completions and get routed into
additionalModelRequestFields on Converse (which has no typed inferenceConfig
slot for them).

Run:  python harness/test_providers.py   (exit 0 = all passed)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from providers import BedrockConverse, _drop_unsupported_sampling, _is_frontier_openai


class TestIsFrontierOpenAI(unittest.TestCase):
    def test_matches_gpt5_and_gpt6_with_or_without_profile_prefix(self):
        for mid in (
            "openai.gpt-5.6-terra",
            "us.openai.gpt-5.6-terra",
            "global.openai.gpt-6-astra",
        ):
            self.assertTrue(_is_frontier_openai(mid), mid)

    def test_does_not_match_gpt_oss_or_other_providers(self):
        for mid in ("openai.gpt-oss-120b-1:0", "anthropic.claude-opus-4-6-v1", ""):
            self.assertFalse(_is_frontier_openai(mid), mid)


class TestDropUnsupportedSampling(unittest.TestCase):
    def test_drops_temperature_and_top_p_for_frontier_models(self):
        params = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 64}
        cleaned = _drop_unsupported_sampling("us.openai.gpt-5.6-terra", params)
        self.assertNotIn("temperature", cleaned)
        self.assertNotIn("top_p", cleaned)
        self.assertEqual(cleaned["max_tokens"], 64)

    def test_leaves_reasoning_effort_and_verbosity_untouched(self):
        params = {"temperature": 0.0, "reasoning_effort": "high", "verbosity": "low"}
        cleaned = _drop_unsupported_sampling("us.openai.gpt-5.6-terra", params)
        self.assertEqual(cleaned["reasoning_effort"], "high")
        self.assertEqual(cleaned["verbosity"], "low")
        self.assertNotIn("temperature", cleaned)

    def test_passes_through_unchanged_for_non_frontier_models(self):
        params = {"temperature": 0.2, "top_p": 0.9}
        cleaned = _drop_unsupported_sampling("openai.gpt-oss-120b-1:0", params)
        self.assertEqual(cleaned, params)

    def test_does_not_mutate_input_dict(self):
        params = {"temperature": 0.2}
        _drop_unsupported_sampling("us.openai.gpt-5.6-terra", params)
        self.assertEqual(params, {"temperature": 0.2})


class TestConverseParamMapping(unittest.TestCase):
    def test_inference_config_excludes_reasoning_params(self):
        params = {"temperature": 0.2, "max_tokens": 64, "reasoning_effort": "high", "verbosity": "low"}
        cfg = BedrockConverse._to_inference_config(params)
        self.assertEqual(cfg, {"temperature": 0.2, "maxTokens": 64})

    def test_additional_fields_carries_reasoning_effort_and_verbosity(self):
        params = {"reasoning_effort": "high", "verbosity": "low", "max_tokens": 64}
        fields = BedrockConverse._to_additional_fields(params)
        self.assertEqual(fields, {"reasoning_effort": "high", "verbosity": "low"})

    def test_additional_fields_empty_when_absent(self):
        self.assertEqual(BedrockConverse._to_additional_fields({"temperature": 0.2}), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
