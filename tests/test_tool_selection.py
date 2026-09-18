"""
Tests validating that the agent routes to the correct tool(s) for a set
of sample queries: single-tool, multi-tool, and no-tool cases.

The LLM call itself is mocked here (see the design note in the README):
these tests check our orchestration/routing logic, not Groq's model
quality. The one true end-to-end test with a real LLM call lives in
test_integration.py.
"""

import json
from unittest.mock import patch, MagicMock

from src.agent.orchestrator import run_agent


def _fake_tool_call(call_id, name, arguments: dict):
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = name
    tool_call.function.arguments = json.dumps(arguments)
    return tool_call


def _fake_response(tool_calls=None, content=None):
    """Builds a fake object shaped like Groq's chat.completions.create() response."""
    message = MagicMock()
    message.tool_calls = tool_calls
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


@patch("src.agent.orchestrator.client")
def test_single_tool_selection_rag_only(mock_client):
    # First call: model decides to call the RAG tool.
    # Second call: model has the tool result and gives a final answer.
    first_call = _fake_response(
        tool_calls=[_fake_tool_call("call_1", "search_destination_guide",
                                     {"query": "visa requirements for Barcelona"})]
    )
    second_call = _fake_response(content="You do not need a visa for short stays.")
    mock_client.chat.completions.create.side_effect = [first_call, second_call]

    answer = run_agent("What is the visa requirement for entering Barcelona?")

    assert "visa" in answer.lower()
    assert mock_client.chat.completions.create.call_count == 2


@patch("src.agent.orchestrator.client")
def test_multi_tool_selection_rag_and_weather(mock_client):
    first_call = _fake_response(
        tool_calls=[_fake_tool_call("call_1", "search_destination_guide",
                                     {"query": "packing tips for Reykjavik"})]
    )
    second_call = _fake_response(
        tool_calls=[_fake_tool_call("call_2", "get_weather_forecast",
                                     {"city": "Reykjavik", "date_or_month": "February"})]
    )
    third_call = _fake_response(content="Pack warm waterproof layers.")
    mock_client.chat.completions.create.side_effect = [first_call, second_call, third_call]

    answer = run_agent("What should I pack for Reykjavik in February?")

    assert "pack" in answer.lower()
    assert mock_client.chat.completions.create.call_count == 3


@patch("src.agent.orchestrator.client")
def test_no_tool_selection_out_of_scope(mock_client):
    # Model decides no tool is needed and directly refuses.
    only_call = _fake_response(
        content="I can't help with booking flights, but I can share destination info."
    )
    mock_client.chat.completions.create.side_effect = [only_call]

    answer = run_agent("Can you book my flight to Bangkok?")

    assert "book" in answer.lower() or "can't" in answer.lower()
    assert mock_client.chat.completions.create.call_count == 1


@patch("src.agent.orchestrator.client")
def test_empty_query_never_calls_llm(mock_client):
    answer = run_agent("")
    assert "empty" in answer.lower() or "question" in answer.lower()
    mock_client.chat.completions.create.assert_not_called()


@patch("src.agent.orchestrator.client")
def test_llm_failure_is_handled_gracefully(mock_client):
    mock_client.chat.completions.create.side_effect = Exception("network error")

    answer = run_agent("What is the weather in Tokyo in July?")

    assert "sorry" in answer.lower() or "problem" in answer.lower()