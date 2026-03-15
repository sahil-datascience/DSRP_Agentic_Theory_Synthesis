from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
from utils.parse_llm_json import parse_llm_json
import json
from utils.dsrp_state import DSRPState


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text", "")
                if isinstance(text_value, str):
                    parts.append(text_value)
                else:
                    parts.append(str(text_value))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _normalize_evidence_item(item):
    item = item if isinstance(item, dict) else {}
    return {
        "id": item.get("id", ""),
        "page": item.get("page", ""),
        "section": item.get("section", ""),
        "direct_quote": item.get("direct_quote", ""),
    }


def _extract_evidence_list(evidence_json) -> list:
    if isinstance(evidence_json, list):
        return [_normalize_evidence_item(item) for item in evidence_json]

    if isinstance(evidence_json, dict):
        for key in ["evaluation_evidence", "candidate_evidence", "evidence"]:
            value = evidence_json.get(key, [])
            if isinstance(value, list):
                return [_normalize_evidence_item(item) for item in value]

    return []


def _to_evidence_map(evidence_list: list) -> dict:
    evidence_map = {}
    for item in evidence_list:
        evidence_id = item.get("id", "")
        if evidence_id != "":
            evidence_map[str(evidence_id)] = item
    return evidence_map


def _normalize_task(task_payload, fallback_label="", fallback_reasoning="", evidence_lookup=None, fallback_evidence=None):
    task_payload = task_payload if isinstance(task_payload, dict) else {}
    raw_evidence = task_payload.get("raw_evidence", [])
    if not isinstance(raw_evidence, list):
        raw_evidence = []

    normalized_evidence = []
    for item in raw_evidence:
        if isinstance(item, dict):
            normalized_item = _normalize_evidence_item(item)
            if normalized_item.get("direct_quote", ""):
                normalized_evidence.append(normalized_item)
                continue

            evidence_id = normalized_item.get("id", "")
            if evidence_lookup and str(evidence_id) in evidence_lookup:
                normalized_evidence.append(evidence_lookup[str(evidence_id)])
            else:
                normalized_evidence.append(normalized_item)
        else:
            evidence_id = str(item)
            if evidence_lookup and evidence_id in evidence_lookup:
                normalized_evidence.append(evidence_lookup[evidence_id])

    label_value = task_payload.get("label", fallback_label)
    label_text = str(label_value).strip().lower()
    no_evidence_expected = label_text in {"not reported", "not_reported", "not applicable", ""}

    if not normalized_evidence and not no_evidence_expected and fallback_evidence:
        normalized_evidence = fallback_evidence[:2]

    return {
        "label": label_value,
        "raw_evidence": normalized_evidence,
        "reasoning": task_payload.get("reasoning", fallback_reasoning),
    }


def _normalize_foundational_output(audit_json: dict, evidence_list: list) -> dict:
    if not isinstance(audit_json, dict):
        audit_json = {}

    overall_reasoning = audit_json.get("overall_reasoning", "")
    evidence_lookup = _to_evidence_map(evidence_list)

    audit_json["evaluation_strategy"] = _normalize_task(
        audit_json.get("evaluation_strategy", {}),
        fallback_label="",
        fallback_reasoning=overall_reasoning,
        evidence_lookup=evidence_lookup,
        fallback_evidence=evidence_list,
    )
    audit_json["validation_procedure"] = _normalize_task(
        audit_json.get("validation_procedure", {}),
        fallback_label="",
        fallback_reasoning=overall_reasoning,
        evidence_lookup=evidence_lookup,
        fallback_evidence=evidence_list,
    )
    audit_json["effect_size_reported"] = _normalize_task(
        audit_json.get("effect_size_reported", {}),
        fallback_label="",
        fallback_reasoning=overall_reasoning,
        evidence_lookup=evidence_lookup,
        fallback_evidence=evidence_list,
    )
    audit_json["assumption_checks_reported"] = _normalize_task(
        audit_json.get("assumption_checks_reported", {}),
        fallback_label="",
        fallback_reasoning=overall_reasoning,
        evidence_lookup=evidence_lookup,
        fallback_evidence=evidence_list,
    )

    if "confidence_score" not in audit_json:
        audit_json["confidence_score"] = 0.0

    if "overall_reasoning" not in audit_json:
        audit_json["overall_reasoning"] = ""

    if "audit_commentary" not in audit_json:
        audit_json["audit_commentary"] = ""

    return audit_json

# =====================================================

# FOUNDATIONAL EVALUATION METRICS NODE

# =====================================================

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
        retriever_prompt.format_messages(
            input=context_text
        )
    )

    evidence_json = parse_llm_json(
        _message_content_to_text(evidence_response.content)
    )
    evidence_list = _extract_evidence_list(evidence_json)

    # =====================================================
    # 3️⃣ LOAD MODELLING OUTPUT (CRITICAL CONTEXT)
    # =====================================================

    modelling_output = state["dsrp_outputs"].get("modelling", {})

    foundational_paradigm = modelling_output.get(
        "foundational_paradigm"
    )

    ml_learning_type = modelling_output.get(
        "ml_learning_type", [])

    ml_problem_type = modelling_output.get(
        "ml_problem_type",[])
    
    primary_learning_type = ml_learning_type[0] if isinstance(ml_learning_type, list) else ml_learning_type
    primary_problem_type = ml_problem_type[0] if isinstance(ml_problem_type, list) else ml_problem_type
    
    # =====================================================
    # 4️⃣ METRICS CLASSIFICATION
    # =====================================================

    classifier_prompt = load_yaml_prompt(
        f"{base_path}/classifier.yaml"
    )

    classifier_input = {
        "modelling_context": {
            "foundational_paradigm": foundational_paradigm,
            "ml_learning_type": primary_learning_type,
            "ml_problem_type": primary_problem_type
        },
        "evidence": evidence_list
    }

    classification_response = llm.invoke(
        classifier_prompt.format_messages(
            input=json.dumps(classifier_input)
        )
    )

    classification_json = parse_llm_json(
        _message_content_to_text(classification_response.content)
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
            "ml_learning_type": ml_learning_type,
            "ml_problem_type": ml_problem_type
        },
        "evidence": evidence_list,
        "classification": classification_json
    }

    audit_response = llm.invoke(
        auditor_prompt.format_messages(
            input=json.dumps(audit_input)
        )
    )

    audit_json = parse_llm_json(
        _message_content_to_text(audit_response.content)
    )

    audit_json = _normalize_foundational_output(audit_json, evidence_list)

    # =====================================================
    # 6️⃣ STORE RESULT
    # =====================================================

    state["dsrp_outputs"]["evaluation_metrics_foundational"] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}
    
