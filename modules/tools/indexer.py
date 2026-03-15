import os
from pathlib import Path
from typing import Optional

from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class TheoryIndexerConfigError(Exception):
    pass


class PaperIndexer:

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):

        if not collection_name:
            raise TheoryIndexerConfigError(
                "collection_name must be provided."
            )

        if not persist_directory:
            raise TheoryIndexerConfigError(
                "persist_directory must be provided."
            )

        if not embedding_model:
            raise TheoryIndexerConfigError(
                "embedding_model must be provided."
            )

        self._vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(
                model=embedding_model
            ),
            persist_directory=persist_directory
        )

    # -------------------------
    # Public API
    # -------------------------

    def index_folder(self, folder_path: str):

        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))

        print(f"🔍 Found {len(pdf_files)} PDF files.")

        for pdf in pdf_files:
            self._index_single_file(str(pdf))

        print("🎉 Indexing complete.")

    # -------------------------
    # Internal Methods
    # -------------------------

    def _paper_exists(self, paper_id: str) -> bool:
        results = self._vectorstore.get(where={"paper_id": paper_id})
        return len(results["ids"]) > 0

    def _index_single_file(self, filepath: str):

        filename = os.path.basename(filepath)
        paper_id = filename

        print(f"\n📄 Checking: {filename}")

        if self._paper_exists(paper_id):
            print("⚠️ Already indexed. Skipping.")
            return

        print("🔄 Running Docling...")

        loader = DoclingLoader(
            file_path=filepath,
            chunker=HybridChunker()
        )

        docs = loader.load()

        documents, ids = self._convert_docs(docs, paper_id)

        self._vectorstore.add_documents(documents, ids=ids)
        self._vectorstore.persist()

        print("✅ Indexed successfully.")

    def _convert_docs(self, doc_splits, paper_id):

        documents = []
        ids = []

        for i, doc in enumerate(doc_splits):

            dl_meta = doc.metadata.get("dl_meta", {})
            headings = dl_meta.get("headings", [])
            doc_items = dl_meta.get("doc_items", [])

            page_no = None
            if doc_items:
                prov = doc_items[0].get("prov", [])
                if prov:
                    page_no = prov[0].get("page_no")

            chunk_id = f"{paper_id}_chunk_{i}"

            metadata = {
                "paper_id": paper_id,
                "section_heading": headings[0] if headings else None,
                "page_no": page_no,
                "modality": "text"
            }

            documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
            )

            ids.append(chunk_id)

        return documents, ids
