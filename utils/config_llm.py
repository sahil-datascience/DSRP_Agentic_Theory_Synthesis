
import os

from langchain_openai import ChatOpenAI


def set_llm(model: str | None = None, temperature: float = 0.5):
    # Priority: explicit arg > env var > project default.
    selected_model = model or os.getenv("DSRP_LLM_MODEL") or "gpt-4o-mini"

    return ChatOpenAI(
        model=selected_model,
        temperature=temperature,
    )

