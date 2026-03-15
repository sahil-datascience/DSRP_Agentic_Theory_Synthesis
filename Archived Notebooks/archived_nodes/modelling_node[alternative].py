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

    # =====================================================
    # 1️⃣ GLOBAL VECTOR RETRIEVAL
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
    # 2️⃣ FOUNDATIONAL MODELLING CLASSIFICATION
    # =====================================================

    foundational_prompt = load_yaml_prompt(
        f"{base_path}/foundational_classifier.yaml"
    )

    foundational_response = llm.invoke(
        foundational_prompt.format_messages(input=context_text)
    )

    foundational_json = parse_llm_json(foundational_response.content)

    # =====================================================
    # 3️⃣ MACHINE LEARNING CLASSIFICATION
    # =====================================================

    ml_json = {
        "ml_learning_type": "not_applicable",
        "ml_problem_type": [],
        "deep_learning_used": False
    }

    if foundational_json.get("foundational_paradigm") in [
        "Machine Learning",
        "Mixed"
    ]:

        ml_prompt = load_yaml_prompt(
            f"{base_path}/ml_learning_classifier.yaml"
        )

        ml_response = llm.invoke(
            ml_prompt.format_messages(input=context_text)
        )

        ml_json = parse_llm_json(ml_response.content)

    # =====================================================
    # 4️⃣ SPECIALISED PARADIGM CLASSIFICATION
    # =====================================================

    specialised_prompt = load_yaml_prompt(
        f"{base_path}/specialised_classifier.yaml"
    )

    specialised_response = llm.invoke(
        specialised_prompt.format_messages(input=context_text)
    )

    specialised_json = parse_llm_json(specialised_response.content)

    # =====================================================
    # 5️⃣ GLOBAL MODELLING AUDIT
    # =====================================================

    auditor_prompt = load_yaml_prompt(
        f"{base_path}/auditor.yaml"
    )

    audit_response = llm.invoke(
        auditor_prompt.format_messages(
            input=json.dumps({
                "foundational": foundational_json,
                "ml": ml_json,
                "specialised": specialised_json,
                "context": context_text
            })
        )
    )

    audit_json = parse_llm_json(audit_response.content)

    # =====================================================
    # 6️⃣ STORE FINAL OUTPUT
    # =====================================================

    state["dsrp_outputs"]["modelling"] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}