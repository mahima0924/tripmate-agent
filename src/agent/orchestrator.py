"""
Agent / Orchestrator

This is the agentic core of TripMate. It:
  1. Accepts a natural-language user query.
  2. Lets the LLM (via Groq's OpenAI-compatible function-calling) decide
     which tool(s), if any, are relevant -- dynamically, not via
     keyword/if-else routing.
  3. Executes the chosen tool(s) in the order the LLM requests them.
  4. Feeds tool results back to the LLM to synthesize one coherent
     natural-language answer.
  5. Logs every decision and tool call as a visible reasoning trace.

Design note: we deliberately use Groq's raw function-calling API
(no LangChain/LangGraph) so every step of the loop is explicit and
easy to explain/demo -- appropriate for this assessment's scope.
"""

import json
import logging
from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.tools.rag_tool import search_destination_guide
from src.tools.weather_tool import get_weather_forecast

logger = logging.getLogger("tripmate.orchestrator")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------
# Tool schemas exposed to the LLM (OpenAI-compatible function format,
# which Groq supports natively).
# ---------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_destination_guide",
            "description": (
                "Search the destination knowledge base for information about "
                "visa requirements, best time to visit, local customs, packing "
                "tips, or safety notes for a specific city. Use this whenever "
                "the user asks about destination-specific facts or advice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A natural-language question or topic, e.g. "
                            "'visa requirements for Tokyo' or 'is Bangkok safe'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": (
                "Get typical weather conditions (temperature range and "
                "conditions) for a city during a given month or date. Use this "
                "whenever the user asks about weather, temperature, climate, "
                "or when deciding what to pack based on season."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Tokyo'.",
                    },
                    "date_or_month": {
                        "type": "string",
                        "description": (
                            "A month name (e.g. 'December') or a date "
                            "(e.g. '2025-12-10')."
                        ),
                    },
                },
                "required": ["city", "date_or_month"],
            },
        },
    },
]

# Maps tool name -> actual Python callable.
TOOL_REGISTRY = {
    "search_destination_guide": search_destination_guide,
    "get_weather_forecast": get_weather_forecast,
}

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
- If search_destination_guide returns an empty list, this means the \
destination is NOT in the knowledge base (only Tokyo, Barcelona, Bangkok, \
and Reykjavik are supported). In that case, clearly tell the user you don't \
have destination-guide data for that city and name the 4 cities you do \
support. Do not answer the destination-specific question anyway using your \
own knowledge.
- The weather tool works for any real city (it's not limited to the 4 \
supported destinations), so you may still answer pure weather questions for \
other cities -- just not visa/customs/packing/safety questions, which depend \
on the destination guide.
- For packing questions about one of the 4 supported cities, use BOTH the \
destination guide tool AND the weather tool, since good packing advice \
depends on both local tips and the actual season.
- If a request is outside your scope (e.g. booking flights/hotels, payments, \
or anything unrelated to destination info), clearly state that you cannot do \
this. Do not pretend to perform the action.
- If a tool returns an error, say so honestly rather than fabricating an \
answer.
- Keep answers concise, friendly, and directly useful for trip planning.
"""

MAX_TOOL_ITERATIONS = 4  # safety cap against infinite tool-call loops


def run_agent(user_query: str) -> str:
    """
    Runs one full agent turn: takes a user query, lets the LLM decide on
    and execute tool calls (possibly chained), and returns the final
    natural-language answer.
    """
    if not user_query or not user_query.strip():
        logger.warning("Empty query received.")
        return "Please enter a question -- I can't help with an empty request."

    logger.info("=== New query === %r", user_query)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for iteration in range(MAX_TOOL_ITERATIONS):
        logger.debug("Calling LLM (iteration %d)...", iteration + 1)
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return (
                "Sorry, I ran into a problem reaching the language model. "
                "Please try again in a moment."
            )

        choice = response.choices[0]
        message = choice.message

        # No tool calls -> the model has a final answer.
        if not message.tool_calls:
            final_answer = message.content or "I don't have a response for that."
            logger.info("Final answer (no further tools needed): %s", final_answer)
            return final_answer

        # The model wants to call one or more tools -- append its request
        # to the conversation, then execute each requested tool in order.
        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error("Malformed tool arguments from LLM: %s", e)
                tool_result = {"error": "Malformed tool arguments."}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                })
                continue

            logger.info(
                "Reasoning: model chose tool=%s with args=%s",
                tool_name, tool_args
            )

            tool_fn = TOOL_REGISTRY.get(tool_name)
            if tool_fn is None:
                logger.error("Model requested unknown tool: %s", tool_name)
                tool_result = {"error": f"Unknown tool '{tool_name}'."}
            else:
                try:
                    tool_result = tool_fn(**tool_args)
                    logger.info(
                        "Tool call succeeded: tool=%s args=%s result=%s",
                        tool_name, tool_args, tool_result
                    )
                except Exception as e:
                    logger.error(
                        "Tool call failed: tool=%s args=%s error=%s",
                        tool_name, tool_args, e
                    )
                    tool_result = {"error": f"Tool '{tool_name}' failed: {e}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result),
            })

    logger.warning("Max tool iterations reached without a final answer.")
    return (
        "I wasn't able to fully resolve this request after multiple tool "
        "calls. Could you rephrase or narrow your question?"
    )