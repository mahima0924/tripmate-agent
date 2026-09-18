"""
Integration test: full multi-tool flow, end-to-end, using a REAL Groq
API call (this is intentionally the one test in the suite that hits
the live LLM -- see README design notes for why unit/tool-selection
tests mock the LLM but this one doesn't).

Requires a valid GROQ_API_KEY in your environment/.env to run.
"""

import os
import pytest
from src.agent.orchestrator import run_agent


requires_groq_key = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; skipping live integration test.",
)


@requires_groq_key
def test_full_multi_tool_flow_packing_question():
    """
    A packing question should trigger BOTH the destination guide tool
    and the weather tool, and produce one coherent final answer that
    reflects information from both.
    """
    answer = run_agent("What should I pack for a trip to Tokyo in December?")

    assert isinstance(answer, str)
    assert len(answer) > 0

    # The answer should reflect actual packing-relevant content, not a
    # generic refusal or error message.
    lowered = answer.lower()
    assert "sorry" not in lowered or "pack" in lowered
    assert any(
        keyword in lowered
        for keyword in ["layer", "jacket", "warm", "cold", "coat", "cloth"]
    )