
from typing import Annotated, Any, Dict, NotRequired, TypedDict


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge parallel updates for dict-backed state channels."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class DSRPState(TypedDict):
    paper_id: str
    dsrp_outputs: Annotated[Dict[str, Any], merge_dicts]
    collection_name: str
    persist_directory: str
    embedding_model: str
    llm_model: NotRequired[str]
    gatekeeper: NotRequired[Dict[str, Any]]