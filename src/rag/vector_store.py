from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import VECTOR_STORE_PATH


class TextDocument:
    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.page_content = content
        self.metadata = metadata or {}


class VectorStoreManager:
    def __init__(self):
        self.vector_store_path = Path(VECTOR_STORE_PATH)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self._index_file = self.vector_store_path / "documents.json"
        self._vectorizer = TfidfVectorizer()
        self._documents: List[TextDocument] = []
        self._matrix = None
        self._load()

    def create_document(self, content, metadata=None):
        return TextDocument(content=content, metadata=metadata)

    def add_documents(self, documents: List[TextDocument]):
        self._documents.extend(documents)
        self._rebuild_index()
        self._save()

    def add_texts(self, texts, metadatas=None):
        """添加文本列表到向量存储"""

        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(self.create_document(text, metadata))
        self.add_documents(documents)
        return len(documents)

    def similarity_search(self, query: str, top_k: int = 5) -> List[TextDocument]:
        if not self._documents or self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix).flatten()
        indices = sims.argsort()[::-1][:top_k]
        return [self._documents[i] for i in indices]

    def _rebuild_index(self):
        corpus = [doc.page_content for doc in self._documents]
        self._matrix = self._vectorizer.fit_transform(corpus) if corpus else None

    def _save(self):
        payload = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in self._documents
        ]
        self._index_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _load(self):
        if not self._index_file.exists():
            return
        raw = json.loads(self._index_file.read_text(encoding="utf-8"))
        self._documents = [TextDocument(item["content"], item.get("metadata", {})) for item in raw]
        if self._documents:
            self._rebuild_index()
