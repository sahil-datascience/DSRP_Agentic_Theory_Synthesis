

import json
import re

def parse_llm_json(text: str) -> dict:
    s = (text or "").strip()

    # remove markdown fences if present
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()

    # try direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # fix invalid single backslashes: \X -> \\X (only when not a valid JSON escape)
    s2 = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', s)

    try:
        return json.loads(s2)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw:\n{s[:1200]}")