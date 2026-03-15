
# Retrieval tool with minimal provenance information, returning results in JSON format.
from langchain.tools import tool
import json

def make_retriever_tool(retriever):
    @tool
    def retrieve_with_min_provenance_json(query: str) -> str:
        """Return chunks with minimal provenance in JSON."""
        docs = retriever.invoke(query)
        items = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            dl_meta = meta.get("dl_meta", {})
            doc_items = dl_meta.get("doc_items", [])
            prov = (doc_items[0].get("prov") or [{}])[0] if doc_items else {}
            page_no = prov.get("page_no")
            headings = dl_meta.get("headings") or []
            origin = dl_meta.get("origin", {})
            filename = origin.get("filename") or meta.get("source", "unknown")

            items.append({
                "id": i,
                "content": doc.page_content,
                "provenance": {
                    "filename": filename,
                    "page_no": page_no,
                    "headings": headings
                }
            })
        return json.dumps(items, ensure_ascii=False, indent=2)
    return retrieve_with_min_provenance_json