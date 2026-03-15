

from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
from utils.parse_llm_json import parse_llm_json
import json
from utils.dsrp_state import DSRPState

def research_question_node(state: DSRPState):

    dimension_name = "research_question"
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]

    # Load vector query
    vector_config = load_vector_query(
        f"prompts/dsrp/{dimension_name}/vector_query.yaml"
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

    # Format context
    context_text = "\n\n".join(
        f"[Page {d.metadata.get('page_no')} | {d.metadata.get('section_heading')}]\n{d.page_content}"
        for d in docs
    )

    # Evidence extraction
    retriever_prompt = load_yaml_prompt(
        f"prompts/dsrp/{dimension_name}/retriever.yaml"
    )

    # Set up LLM
    llm = set_llm()

    evidence_response = llm.invoke(
        retriever_prompt.format_messages(input=context_text)
    )

    evidence_json = parse_llm_json(evidence_response.content)

    # Classification
    classifier_prompt = load_yaml_prompt(
        f"prompts/dsrp/{dimension_name}/classifier.yaml"
    )

    classification_response = llm.invoke(
        classifier_prompt.format_messages(
            input=json.dumps(evidence_json)
        )
    )

    classification_json = parse_llm_json(classification_response.content)

    # Audit
    auditor_prompt = load_yaml_prompt(
        f"prompts/dsrp/{dimension_name}/auditor.yaml"
    )

    audit_response = llm.invoke(
        auditor_prompt.format_messages(
            input=json.dumps(classification_json)
        )
    )

    audit_json = parse_llm_json(audit_response.content)

    # Store result
    state["dsrp_outputs"][dimension_name] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}