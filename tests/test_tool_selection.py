"""
Tests validating that the agent routes to the correct tool(s) for a set
of sample queries: single-tool, multi-tool, and no-tool cases.

The underlying LangChain agent graph (_agent.invoke) is mocked here, so
these tests check our wrapper logic (run_agent) -- not Groq's model
quality or LangChain's internals. The one true end-to-end test with a
real Groq call lives in test_integration.py.
"""

from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agent.orchestrator import run_agent


def _ai_tool_call_message(tool_name, args, call_id="call_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": call_id}],
    )


def _tool_message(content, call_id="call_1"):
    return ToolMessage(content=content, tool_call_id=call_id)


def _final_ai_message(content):
    return AIMessage(content=content)


@patch("src.agent.orchestrator._agent")
def test_single_tool_selection_rag_only(mock_agent):
    messages = [
        HumanMessage(content="What is the visa requirement for entering Tokyo?"),
        _ai_tool_call_message("search_destination_guide", {"query": "visa for Tokyo"}),
        _tool_message("Tokyo - VISA & ENTRY: visa-free for short stays."),
        _final_ai_message("You do not need a visa for short stays in Tokyo."),
    ]
    mock_agent.invoke.return_value = {"messages": messages}

    answer = run_agent("What is the visa requirement for entering Tokyo?")

    assert "visa" in answer.lower()
    mock_agent.invoke.assert_called_once()


@patch("src.agent.orchestrator._agent")
def test_multi_tool_selection_rag_and_weather(mock_agent):
    messages = [
        HumanMessage(content="What should I pack for Bangkok in August?"),
        _ai_tool_call_message(
            "search_destination_guide", {"query": "packing tips Bangkok"}, call_id="call_1"
        ),
        _tool_message("Bangkok - PACKING TIPS: lightweight clothing.", call_id="call_1"),
        _ai_tool_call_message(
            "get_weather_forecast",
            {"city": "Bangkok", "date_or_month": "August"},
            call_id="call_2",
        ),
        _tool_message(
            "{'temp_range_c': [25.5, 32.4], 'conditions': 'frequent rain likely'}",
            call_id="call_2",
        ),
        _final_ai_message("Pack light, breathable clothing and rain gear."),
    ]
    mock_agent.invoke.return_value = {"messages": messages}

    answer = run_agent("What should I pack for Bangkok in August?")

    assert "pack" in answer.lower()
    mock_agent.invoke.assert_called_once()


@patch("src.agent.orchestrator._agent")
def test_no_tool_selection_out_of_scope(mock_agent):
    messages = [
        HumanMessage(content="Can you book my flight to Tokyo?"),
        _final_ai_message(
            "I can't help with booking flights, but I can share destination info."
        ),
    ]
    mock_agent.invoke.return_value = {"messages": messages}

    answer = run_agent("Can you book my flight to Tokyo?")

    assert "book" in answer.lower() or "can't" in answer.lower()
    mock_agent.invoke.assert_called_once()


@patch("src.agent.orchestrator._agent")
def test_empty_query_never_calls_agent(mock_agent):
    answer = run_agent("")
    assert "empty" in answer.lower() or "question" in answer.lower()
    mock_agent.invoke.assert_not_called()


@patch("src.agent.orchestrator._agent")
def test_agent_failure_is_handled_gracefully(mock_agent):
    mock_agent.invoke.side_effect = Exception("network error")

    answer = run_agent("What is the weather in Tokyo in July?")

    assert "sorry" in answer.lower() or "problem" in answer.lower()