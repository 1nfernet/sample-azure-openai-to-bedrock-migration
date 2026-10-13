# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Responses API researcher agent (explicit tool-call loop), on Bedrock.

A third migration target for ../before/azure_researcher.py, for teams that
already use the OpenAI **Responses API** (`client.responses.create(...)`) — or
are on Azure OpenAI Assistants, which maps conceptually to it — and want to keep
that surface for their agent loop instead of adopting Converse.

  - Staying on Converse / pure boto3?  see bedrock_converse_loop.py.
  - Want a managed AWS agent runtime?  see bedrock_strands_agentcore.py.
  - This file: keep the Responses API, point it at Bedrock, own the loop.

It migrates the same researcher scenario as bedrock_converse_loop.py and returns
the same shape ({"web": [...], "entities": [...], "news": [...]}), so you can
diff the two targets side by side.

Why the `bedrock-mantle` endpoint (not `bedrock-runtime`)? The Responses API is
available on both, but mantle is where the server-side built-in tools live —
this example optionally wires in the mantle-only Web Search tool alongside the
custom function tools. On `bedrock-runtime` the Responses API has no server-side
Web Search. See ../../01_chat_completions/after/bedrock_responses_api.py.

How the loop differs from Converse (what actually changes):

  Converse (bedrock_converse_loop.py)        Responses API (this file)
  ─────────────────────────────────────────  ──────────────────────────────────────
  response["stopReason"] == "tool_use"        output items of type "function_call"
  toolConfig={"tools":[{"toolSpec":...}]}     tools=[{"type":"function","name":...}]
  inputSchema={"json": <schema>}              parameters=<schema> (flat, top-level)
  append {"toolResult":{"toolUseId",...}}     append {"type":"function_call_output",
                                                       "call_id":..., "output":...}
  resend full `messages` each turn            resend running `input` list each turn
                                              (or chain with previous_response_id)
  server prompt via system=[{"text":...}]     instructions="..." (top-level)

Note the tool-call id names differ: Converse pairs toolUse/toolResult by
`toolUseId`; the Responses API pairs a `function_call`'s `call_id` with a
`function_call_output`'s `call_id`.

This file drives the loop **statelessly** — it resends the growing `input` list
each turn (mirroring the Converse example, which must resend `messages` because
Converse is stateless). The Responses API also supports server-side state via
`previous_response_id`; see the note in `_run_tool_loop`.

Web search backing is a stub (same as the Converse example): replace
_web_search() / _entity_search() / _news_search() with Tavily, Brave, etc., or
drop them entirely and rely on the mantle Web Search built-in (set
USE_BEDROCK_WEB_SEARCH=1).

Install:
    pip install openai aws-bedrock-token-generator

Env vars:
    AWS_REGION             (default: us-east-1)
    BEDROCK_MODEL_ID       (default: openai.gpt-5.6-terra — the BARE mantle id;
                            mantle rejects the us.-prefixed inference-profile id)
    USE_BEDROCK_WEB_SEARCH (optional — "1" to add the mantle Web Search built-in
                            tool to the loop)
    TAVILY_API_KEY         (optional — used by the search stubs if set)
    plus standard AWS credentials (profile, env vars, or IAM role).

IAM note: bearer-token auth (provide_token + OpenAI SDK) on mantle needs BOTH
bedrock-mantle:CreateInference AND bedrock-mantle:CallWithBearerToken. See
../least-privilege-role.json.
"""

from __future__ import annotations

import json
import os
from typing import Any

from aws_bedrock_token_generator import provide_token
from openai import OpenAI

REGION = os.environ.get("AWS_REGION", "us-east-1")

# provide_token() mints a short-term Bedrock API key from your current AWS
# credentials — no static key stored in the app or environment.
client = OpenAI(
    base_url=f"https://bedrock-mantle.{REGION}.api.aws/openai/v1",
    api_key=provide_token(region=REGION),
)

# Mantle uses the BARE model id (no `us.` prefix — that's a bedrock-runtime
# inference-profile concept). Override with any mantle model id that supports
# tool use. See ../../../docs/model-mapping.md.
MODEL = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-5.6-terra")

# Optionally add the mantle-only server-side Web Search tool to the loop.
USE_BEDROCK_WEB_SEARCH = os.environ.get("USE_BEDROCK_WEB_SEARCH", "") == "1"


# ── Tools — Responses API shape ───────────────────────────────────────────────
#
# OpenAI function-tool wrapping across the two Bedrock surfaces:
#
#   Converse toolSpec:  {"toolSpec":{"name":N,"description":D,
#                                    "inputSchema":{"json": <schema>}}}
#   Responses API:      {"type":"function","name":N,"description":D,
#                        "parameters": <schema>}
#
# The inner JSON Schema object is IDENTICAL to functions.json / the Converse
# example — only the envelope changes (and the Responses API keeps `name`,
# `description`, `parameters` flat at the top level rather than nested under a
# "function" key as Chat Completions does).

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "An optimal search query"},
        "market": {"type": "string", "description": "Market code, e.g. en-US"},
    },
    "required": ["query"],
    # The Responses API expects additionalProperties:false on strict function
    # tools; include it so the schema is accepted unchanged.
    "additionalProperties": False,
}

FUNCTION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "find_information",
        "description": (
            "Finds information on the web given a query using a search API. "
            "Use for general information — not news or entity lookups."
        ),
        "parameters": _QUERY_SCHEMA,
    },
    {
        "type": "function",
        "name": "find_entities",
        "description": (
            "Finds entities (people, places, things) on the web. "
            "Use for entity lookups, not general information or news."
        ),
        "parameters": _QUERY_SCHEMA,
    },
    {
        "type": "function",
        "name": "find_news",
        "description": (
            "Finds recent news articles on the web given a query. "
            "Use when the user is looking for news."
        ),
        "parameters": _QUERY_SCHEMA,
    },
]


def _tools() -> list[dict[str, Any]]:
    """The tool list sent on every turn: custom functions, plus (optionally) the
    mantle server-side Web Search built-in.

    Web Search is executed server-side by Bedrock — unlike the function tools,
    the model does not hand it back to us to run. We keep `external_web_access`
    False so retrieval stays inside the AWS boundary and needs no extra
    bedrock-websearch:ExternalWebAccess permission (see
    ../../01_chat_completions/after/bedrock_responses_api.py for the full note).
    """
    tools = list(FUNCTION_TOOLS)
    if USE_BEDROCK_WEB_SEARCH:
        tools.append(
            {
                "type": "web_search",
                "search_context_size": "low",
                "external_web_access": False,
            }
        )
    return tools


# ── Search backing (replace with Tavily / Brave / etc. in production) ─────────

def _web_search(query: str, market: str = "en-US") -> list[dict]:
    """General web search stub. Replace with a real search API."""
    try:
        import tavily  # type: ignore

        c = tavily.TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = c.search(query, max_results=5)["results"]
        return [
            {"url": r["url"], "name": r["title"], "description": r["content"]}
            for r in results
        ]
    except Exception:
        # Deterministic offline fallback (e.g. for smoke tests). Neutral and
        # plausible so the model treats it as usable grounding.
        return [
            {
                "url": f"https://example.com/search?q={query.replace(' ', '+')}&r={i}",
                "name": f"{query} — reference {i}",
                "description": (
                    f"Overview of {query}. Sample grounding result {i} of 3 "
                    "returned by the offline search stub."
                ),
            }
            for i in range(1, 4)
        ]


def _entity_search(query: str, market: str = "en-US") -> list[dict]:
    return _web_search(query, market)  # delegate; swap for an entity API


def _news_search(query: str, market: str = "en-US") -> list[dict]:
    try:
        import tavily  # type: ignore

        c = tavily.TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = c.search(query, topic="news", max_results=5)["results"]
        return [
            {"url": r["url"], "name": r["title"], "description": r["content"]}
            for r in results
        ]
    except Exception:
        return _web_search(query, market)


_TOOL_FN = {
    "find_information": lambda inp: _web_search(inp["query"], inp.get("market", "en-US")),
    "find_entities": lambda inp: _entity_search(inp["query"], inp.get("market", "en-US")),
    "find_news": lambda inp: _news_search(inp["query"], inp.get("market", "en-US")),
}


# ── Responses API tool loop ───────────────────────────────────────────────────

INSTRUCTIONS = """\
You are a researcher agent. Use the tools to search, then return ONLY a single
JSON object and nothing else.

Rules:
- Do NOT write any prose, explanation, or <thinking> block.
- Do NOT wrap the JSON in markdown code fences.
- Your entire response must be exactly one JSON object with this structure:

{"web": [{"url": "...", "name": "...", "description": "..."}],
 "entities": [],
 "news": []}

Return 3-5 sources in "web". Begin your response with { and end with }.
"""


def _run_tool_loop(user_content: str, max_iterations: int = 10) -> str:
    """Drive the Responses API tool-call loop explicitly.

    Azure AI Agent Service ran this loop server-side (create_and_process_run);
    here we own it. Each turn:

      1. responses.create(input=running, tools=..., instructions=...)
      2. collect `function_call` items from response.output
      3. if none, we're done — return response.output_text
      4. else run each tool, append the model's output items AND a matching
         `function_call_output` for each call to `running`, and loop

    Stateless style: we resend the growing `input` list each turn, exactly like
    the Converse example resends `messages`. Alternatively you can let the
    server hold the state — pass `previous_response_id=response.id` on the next
    turn and send only the new `function_call_output` items in `input`. Both are
    verified to work on mantle; the stateless form is used here because it makes
    the loop explicit and keeps parity with bedrock_converse_loop.py.
    """
    running: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

    for _ in range(max_iterations):
        response = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            input=running,
            tools=_tools(),
            tool_choice="auto",
        )

        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            return response.output_text

        # Append the model's output items (reasoning + the function_call items)
        # verbatim, then our result for each call — the Responses API pairs a
        # function_call to its function_call_output by `call_id`.
        running.extend(item.to_dict() for item in response.output)
        for fc in function_calls:
            fn = _TOOL_FN.get(fc.name)
            try:
                args = json.loads(fc.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if fn is None:
                output = {"error": f"Unknown tool: {fc.name}"}
            else:
                output = {"results": fn(args)}
            running.append(
                {
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": json.dumps(output),
                }
            )

    # Max iterations hit — make one final call to let the model answer.
    final = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=running,
        tools=_tools(),
    )
    return final.output_text


# ── Public entry point — same contract as bedrock_converse_loop.research() ─────

def research(instructions: str, feedback: str = "No feedback") -> dict:
    """Run the researcher agent via the Responses API.

    Drop-in replacement for the Azure/Converse versions — same return shape:
        {"web": [...], "entities": [...], "news": [...]}
    """
    user_content = instructions
    if feedback and feedback.lower() != "no feedback":
        user_content += f"\n\nFeedback to incorporate: {feedback}"

    final_text = _run_tool_loop(user_content)

    try:
        parsed = json.loads(final_text)
    except json.JSONDecodeError:
        # Model returned extra prose — extract the outermost JSON object.
        start = final_text.find("{")
        end = final_text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError(
                f"researcher returned no JSON object; got: {final_text[:200]!r}"
            )
        parsed = json.loads(final_text[start:end])

    return {
        "web": parsed.get("web", []),
        "entities": parsed.get("entities", []),
        "news": parsed.get("news", []),
    }


if __name__ == "__main__":
    result = research("What are the latest camping trends for winter?")
    print(json.dumps(result, indent=2))
