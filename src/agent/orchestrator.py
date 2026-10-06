"""
Agent / Orchestrator — LangChain version

This is the agentic core of TripMate, built using the LangChain framework
(as requested in review feedback) instead of a raw function-calling loop.

It still does the same 5 things Module 1 requires:
  1. Accepts a natural-language user query.
  2. Lets the LLM decide -- dynamically, per query -- which tool(s) are
     relevant (handled internally by LangChain's agent graph).
  3. Executes the chosen tool(s) in the order the LLM requests them.
  4. Synthesizes one coherent final answer.
  5. Logs every step as a visible reasoning trace.

The underlying tool functions (search_destination_guide,
get_weather_forecast) are unchanged -- only the orchestration layer now
uses LangChain's `create_agent` (built on LangGraph under the hood)
instead of a manual Groq SDK tool-calling loop.
"""

import logging

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.tools.rag_tool import search_destination_guide as _search_destination_guide
from src.tools.weather_tool import get_weather_forecast as _get_weather_forecast

logger = logging.getLogger("tripmate.orchestrator")


# ---------------------------------------------------------------------
# Wrap the existing tool functions as LangChain tools.
# The underlying logic (RAG retrieval, weather lookup) is unchanged --
# this is just the adapter layer LangChain needs.
# ---------------------------------------------------------------------

@tool
def search_destination_guide(query: str) -> str:
    """Search the destination knowledge base for information about visa
    requirements, best time to visit, local customs, packing tips, or
    safety notes for a specific city. Use this whenever the user asks
    about destination-specific facts or advice."""
    results = _search_destination_guide(query)
    if not results:
        return "No matching destination data found for this query."
    return "\n\n".join(results)


@tool
def get_weather_forecast(city: str, date_or_month: str) -> str:
    """Get typical weather conditions (temperature range and conditions)
    for a city during a given month or date. Use this whenever the user
    asks about weather, temperature, climate, or when deciding what to
    pack based on season."""
    result = _get_weather_forecast(city, date_or_month)
    return str(result)


TOOLS = [search_destination_guide, get_weather_forecast]

SYSTEM_PROMPT = """You are TripMate, an AI travel assistant. You help travelers \
with questions about destinations: visa requirements, weather, packing advice, \
safety, and local customs, using the tools available to you.

Your destination knowledge base ONLY covers these 4 cities: Tokyo, Barcelona, \
Bangkok, and Reykjavik.

Rules:
- Only use information returned by your tools. Never invent visa rules, \
destination facts, local customs, or safety information from your own \
general knowledge -- even if you happen to know real facts about a place. \
The destination guide tool is the only allowed source for that information.
- If search_destination_guide returns "No matching destination data found",
this means the destination is NOT in the knowledge base (only Tokyo, \
Barcelona, Bangkok, and Reykjavik are supported). In that case, clearly \
tell the user you don't have destination-guide data for that city and name \
the 4 cities you do support. Do not answer the destination-specific \
question anyway using your own knowledge.
- The weather tool works for any real city (it's not limited to the 4 \
supported destinations), so you may still answer pure weather questions \
for other cities -- just not visa/customs/packing/safety questions, which \
depend on the destination guide.
- For packing questions about one of the 4 supported cities, use BOTH the \
destination guide tool AND the weather tool, since good packing advice \
depends on both local tips and the actual season.
- If a request is outside your scope (e.g. booking flights/hotels, \
payments, or anything unrelated to destination info), clearly state that \
you cannot do this. Do not pretend to perform the action.
- If a tool returns an error, say so honestly rather than fabricating an \
answer.
- Keep answers concise, friendly, and directly useful for trip planning.
"""

# Build once at import time, reused across calls.
_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.2)

_agent = create_agent(
    model=_llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


def _log_reasoning_trace(messages):
    """Walk the agent's message history and log each tool call and result,
    giving the 'visible reasoning trace' Module 1 requires."""
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                logger.info(
                    "Reasoning: agent chose tool=%s with args=%s",
                    call["name"], call["args"]
                )
        elif isinstance(msg, ToolMessage):
            logger.info(
                "Tool call result (tool_call_id=%s): %s",
                msg.tool_call_id, msg.content
            )


def run_agent(user_query: str) -> str:
    """
    Runs one full agent turn using LangChain's agent graph: takes a user
    query, lets the LLM decide on and execute tool calls (possibly
    chained), and returns the final natural-language answer.
    """
    if not user_query or not user_query.strip():
        logger.warning("Empty query received.")
        return "Please enter a question -- I can't help with an empty request."

    logger.info("=== New query === %r", user_query)

    try:
        result = _agent.invoke({"messages": [HumanMessage(content=user_query)]})
        messages = result.get("messages", [])

        _log_reasoning_trace(messages)

        # The final answer is the content of the last AIMessage.
        final_answer = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

        if not final_answer:
            final_answer = "I don't have a response for that."

        logger.info("Final answer: %s", final_answer)
        return final_answer

    except Exception as e:
        logger.error("Agent execution failed: %s", e)
        return (
            "Sorry, I ran into a problem processing your request. "
            "Please try again in a moment."
        )