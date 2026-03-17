# Import necessary libraries and modules
from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
from utils.parse_llm_json import parse_llm_json
import json
from utils.dsrp_state import DSRPState


def modelling_node(state: DSRPState):

    base_path = "prompts/dsrp/modelling"
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]
    llm = set_llm()

    foundational_path = f"{base_path}/foundational"
    specialised_path = f"{base_path}/specialised"

    # =====================================================
    # 1️⃣ FOUNDATIONAL VECTOR RETRIEVAL
    # =====================================================

    vector_config = load_vector_query(
        f"{foundational_path}/vector_query.yaml"
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
    # 2️⃣ FOUNDATIONAL EVIDENCE EXTRACTION
    # =====================================================

    retriever_prompt = load_yaml_prompt(
        f"{foundational_path}/retriever.yaml"
    )

    evidence_response = llm.invoke(
        retriever_prompt.format_messages(input=context_text)
    )

    evidence_json = parse_llm_json(evidence_response.content)

    evidence_json["modelling_evidence"] = [
        e for e in evidence_json["modelling_evidence"]
        if e.get("method_used", True)
    ]

    # =====================================================
    # 3️⃣ FOUNDATIONAL CLASSIFICATION
    # =====================================================

    foundational_prompt = load_yaml_prompt(
        f"{foundational_path}/foundational_classifier.yaml"
    )

    foundational_response = llm.invoke(
        foundational_prompt.format_messages(
            input=json.dumps(evidence_json)
        )
    )

    foundational_json = parse_llm_json(foundational_response.content)

    # =====================================================
    # 4️⃣ ML LEARNING CLASSIFICATION
    # =====================================================

    ml_json = {
        "ml_learning_type": [],
        "ml_problem_type": [],
        "deep_learning_used": False
    }

    if foundational_json.get("foundational_paradigm") in [
        "Machine Learning",
        "Mixed"
    ]:

        ml_prompt = load_yaml_prompt(
            f"{foundational_path}/ml_learning_classifier.yaml"
        )

        ml_response = llm.invoke(
            ml_prompt.format_messages(
                input=json.dumps(evidence_json)
            )
        )

        ml_json = parse_llm_json(ml_response.content)

    # =====================================================
    # 5️⃣ FOUNDATIONAL AUDIT
    # =====================================================

    foundational_audit_prompt = load_yaml_prompt(
        f"{foundational_path}/auditor.yaml"
    )

    foundational_audit_response = llm.invoke(
        foundational_audit_prompt.format_messages(
            input=json.dumps({
                "foundational": foundational_json,
                "ml_details": ml_json,
                "evidence": evidence_json
            })
        )
    )

    foundational_audit_json = parse_llm_json(
        foundational_audit_response.content
    )

    # =====================================================
    # 6️⃣ SPECIALISED VECTOR RETRIEVAL
    # =====================================================

    vector_config = load_vector_query(
        f"{specialised_path}/vector_query.yaml"
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

    specialised_context = "\n\n".join(
        f"[Page {d.metadata.get('page_no')} | {d.metadata.get('section_heading')}]\n{d.page_content}"
        for d in docs
    )

    # =====================================================
    # 7️⃣ SPECIALISED EVIDENCE EXTRACTION
    # =====================================================

    retriever_prompt = load_yaml_prompt(
        f"{specialised_path}/retriever.yaml"
    )

    specialised_evidence_response = llm.invoke(
        retriever_prompt.format_messages(
            input=specialised_context
        )
    )

    specialised_evidence_json = parse_llm_json(
        specialised_evidence_response.content
    )

    specialised_evidence_json["modelling_evidence"] = [
        e for e in specialised_evidence_json["modelling_evidence"]
        if e.get("method_used", True)
    ]

    # =====================================================
    # 8️⃣ SPECIALISED CLASSIFICATION
    # =====================================================

    specialised_prompt = load_yaml_prompt(
        f"{specialised_path}/specialised_classifier.yaml"
    )

    specialised_response = llm.invoke(
        specialised_prompt.format_messages(
            input=json.dumps(specialised_evidence_json)
        )
    )

    specialised_json = parse_llm_json(
        specialised_response.content
    )

    # =====================================================
    # 9️⃣ SPECIALISED AUDIT
    # =====================================================

    specialised_audit_prompt = load_yaml_prompt(
        f"{specialised_path}/auditor.yaml"
    )

    specialised_audit_response = llm.invoke(
        specialised_audit_prompt.format_messages(
            input=json.dumps({
                "specialised": specialised_json,
                "evidence": specialised_evidence_json
            })
        )
    )

    specialised_audit_json = parse_llm_json(
        specialised_audit_response.content
    )

    # =====================================================
    # 🔟 GLOBAL MODELLING AUDIT
    # =====================================================

    global_audit_prompt = load_yaml_prompt(
        f"{base_path}/auditor.yaml"
    )

    audit_response = llm.invoke(
        global_audit_prompt.format_messages(
            input=json.dumps({
                "foundational": foundational_audit_json,
                "specialised": specialised_audit_json
            })
        )
    )

    audit_json = parse_llm_json(audit_response.content)

    # =====================================================
    # 1️⃣1️⃣ STORE FINAL OUTPUT
    # =====================================================

    state["dsrp_outputs"]["modelling"] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}