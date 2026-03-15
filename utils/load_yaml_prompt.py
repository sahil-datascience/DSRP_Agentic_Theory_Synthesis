
import yaml
from langchain_core.prompts import ChatPromptTemplate

def load_yaml_prompt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return ChatPromptTemplate.from_messages([
        ("system", config["instructions"]),
        ("human", "{{input}}")
    ],
    template_format="jinja2") # Escape curly braces in the prompt template

