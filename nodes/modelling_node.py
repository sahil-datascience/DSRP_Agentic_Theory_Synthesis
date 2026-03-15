from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
from utils.parse_llm_json import parse_llm_json
import json
from utils.dsrp_state import DSRPState

#foundational metrics node
def evaluation_metrics_foundational_node(state: DSRPState):

    base_path = "prompts/dsrp/evaluation/metrics/foundational"
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]
    llm = set_llm()

    # =====================================================
    # 1️⃣ VECTOR RETRIEVAL
    # =====================================================

    vector_config = load_vector_query(
        f"{base_path}/vector_query.yaml"
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

    # =====================================================
    # 2️⃣ EVIDENCE EXTRACTION
    # =====================================================

    retriever_prompt = load_yaml_prompt(
        f"{base_path}/retriever.yaml"
    )

    evidence_response = llm.invoke(
        retriever_prompt.format_messages(input=context_text)
    )

    evidence_json = parse_llm_json(evidence_response.content)

    # =====================================================
    # 3️⃣ LOAD MODELLING OUTPUT (CRITICAL CONTEXT)
    # =====================================================

    modelling_output = state["dsrp_outputs"].get("modelling", {})

    foundational_paradigm = modelling_output.get("foundational_paradigm")

    ml_learning_type = modelling_output.get("ml_learning_type")

    # =====================================================
    # 4️⃣ METRICS CLASSIFICATION
    # =====================================================

    classifier_prompt = load_yaml_prompt(
        f"{base_path}/classifier.yaml"
    )

    classifier_input = {
        "modelling_context": {
            "foundational_paradigm": foundational_paradigm,
            "ml_learning_type": ml_learning_type
        },
        "evidence": evidence_json
    }

    classification_response = llm.invoke(
        classifier_prompt.format_messages(
            input=json.dumps(classifier_input)
        )
    )

    classification_json = parse_llm_json(
        classification_response.content
    )

    # =====================================================
    # 5️⃣ AUDIT VALIDATION
    # =====================================================

    auditor_prompt = load_yaml_prompt(
        f"{base_path}/auditor.yaml"
    )

    audit_input = {
        "modelling_context": {
            "foundational_paradigm": foundational_paradigm,
            "ml_learning_type": ml_learning_type
        },
        "classification": classification_json
    }

    audit_response = llm.invoke(
        auditor_prompt.format_messages(
            input=json.dumps(audit_input)
        )
    )

    audit_json = parse_llm_json(audit_response.content)

    # =====================================================
    # 6️⃣ STORE RESULT
    # =====================================================

    state["dsrp_outputs"]["evaluation_metrics_foundational"] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}