# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — LangChain agent backed by Amazon Bedrock (ChatBedrockConverse).

The connector swap: langchain_openai.AzureChatOpenAI -> langchain_aws
.ChatBedrockConverse. Your surrounding LangChain code (chains, agents, tools,
.invoke()) is unchanged — you only change how the LLM object is built.

ChatBedrockConverse uses the Converse API under the hood, so it works across all
Bedrock models. Auth is standard AWS credentials; there's no endpoint, key, or
api_version to set.

Frontier GPT reasoning models (openai.gpt-5.x, gpt-6) accept only the default
temperature and reject an explicit value via Converse with a ValidationException
("This model doesn't support the temperature field"). Guard the hardcoded
temperature the same way harness/providers.py's _drop_unsupported_sampling does,
so this example keeps working if BEDROCK_MODEL_ID points at one of those models.
Use reasoning_effort / verbosity instead to control reasoning depth and reply
length on frontier models — see docs/model-mapping.md#reasoning-models-replacing-temperaturetop_p.

Install:
    pip install langchain-aws

Env vars:
    AWS_REGION (default us-east-1), BEDROCK_MODEL_ID (default openai.gpt-oss-120b-1:0),
    plus standard AWS credentials.
"""

import os

from langchain_aws import ChatBedrockConverse

MODEL = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0")


def _is_frontier_openai(model_id: str) -> bool:
    """True for OpenAI frontier models (GPT-5.x / GPT-6) on Bedrock.

    Matches ids like openai.gpt-5.6-terra and us.openai.gpt-6-astra, with or
    without a us./global. inference-profile prefix. These models accept only
    the default temperature and reject an explicit value.
    """
    mid = (model_id or "").lower()
    return "openai.gpt-5" in mid or "openai.gpt-6" in mid


_llm_kwargs = {
    "model": MODEL,
    "region_name": os.environ.get("AWS_REGION", "us-east-1"),
}
if not _is_frontier_openai(MODEL):
    _llm_kwargs["temperature"] = 0.2

llm = ChatBedrockConverse(**_llm_kwargs)


def run(prompt: str) -> str:
    response = llm.invoke(prompt)
    return _text(response.content)


def _text(content) -> str:
    """Extract plain text from a LangChain message.

    With a reasoning model (e.g. OpenAI gpt-oss), ChatBedrockConverse returns
    `content` as a list of blocks (reasoning + text) rather than a plain string.
    Handle both shapes.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


if __name__ == "__main__":
    print(run("In one sentence, what is Amazon Bedrock?"))
