

from langchain_openai import ChatOpenAI

def set_llm():

    return ChatOpenAI(
        model="gpt-4o-mini",   # deterministic & cost-efficient
        temperature=0
    )

