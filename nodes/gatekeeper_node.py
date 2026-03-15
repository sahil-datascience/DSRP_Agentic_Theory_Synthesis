from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
from utils.parse_llm_json import parse_llm_json
import json

from utils.dsrp_state import DSRPState

def gatekeeper_node(state: DSRPState) -> dict:
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]
    llm = set_llm()
    
    # 1️⃣ Load vector query
    vector_config = load_vector_query(
        "prompts/ds_gatekeeper/vector_query.yaml"
    )

    retriever_tool = PaperRetriever(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model
    ).for_paper(
        state["paper_id"],
        k=vector_config["k"]
    )

    docs = retriever_tool.invoke(vector_config["query"])

    context_text = "\n\n".join(
        f"[Page {d.metadata.get('page_no')} | {d.metadata.get('section_heading')}]\n{d.page_content}"
        for d in docs
    )

    # 2️⃣ Evidence extraction
    retriever_prompt = load_yaml_prompt(
        "prompts/ds_gatekeeper/retriever.yaml"
    )

    evidence_response = llm.invoke(
        retriever_prompt.format_messages(input=context_text)
    )

    evidence_json = json.loads(evidence_response.content)

    # 3️⃣ Classification
    classifier_prompt = load_yaml_prompt(
        "prompts/ds_gatekeeper/classifier.yaml"
    )

    classification_response = llm.invoke(
        classifier_prompt.format_messages(
            input=json.dumps(evidence_json)
        )
    )

    classification_json = json.loads(classification_response.content)

    # 4️⃣ Audit
    auditor_prompt = load_yaml_prompt(
        "prompts/ds_gatekeeper/auditor.yaml"
    )

    audit_response = llm.invoke(
        auditor_prompt.format_messages(
            input=json.dumps(classification_json)
        )
    )

    audit_json = json.loads(audit_response.content)

    return {
        "gatekeeper": audit_json
    }