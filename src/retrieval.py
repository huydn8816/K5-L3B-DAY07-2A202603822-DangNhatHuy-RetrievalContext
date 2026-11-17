"""Local dense, lexical, and hybrid retrieval for the Day 7 corpus."""
import json
from pathlib import Path
from typing import Any, Callable

from starter.baseline import TfidfIndex


_BUILTIN_FIELDS = {"source_id", "source", "page"}


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()


def _chroma_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    values = {
        "source_id": chunk["source_id"],
        "source": chunk["source"],
        "page": chunk["page"],
        "metadata_json": json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
    }
    values.update({
        f"meta_{key}": value
        for key, value in chunk.get("metadata", {}).items()
        if isinstance(value, (str, int, float, bool))
    })
    return values


def _where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    clauses = []
    for key, value in filters.items():
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"filter '{key}' phải có giá trị scalar")
        field = key if key in _BUILTIN_FIELDS else f"meta_{key}"
        clauses.append({field: {"$eq": value}})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class DenseRetriever:
    def __init__(self, collection: Any, embed_query: Callable[[list[str]], list[list[float]]]):
        self.collection = collection
        self.embed_query = embed_query

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            raise ValueError("query phải khác rỗng và top_k phải lớn hơn 0")
        vectors = self.embed_query([query])
        response = self.collection.query(
            query_embeddings=vectors,
            n_results=top_k,
            where=_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for chunk_id, text, metadata, distance in zip(
                response["ids"][0], response["documents"][0],
                response["metadatas"][0], response["distances"][0]):
            results.append({
                "id": chunk_id,
                "text": text,
                "source_id": metadata["source_id"],
                "source": metadata["source"],
                "page": metadata["page"],
                "metadata": json.loads(metadata.get("metadata_json", "{}")),
                "score": 1.0 - distance,
                "strategy": "dense",
            })
        return results


class LexicalRetriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.index = TfidfIndex(chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            raise ValueError("query phải khác rỗng và top_k phải lớn hơn 0")
        results = self.index.search(query, top_k=len(self.chunks))
        if filters:
            results = [
                chunk for chunk in results
                if all(
                    (chunk.get(key) if key in _BUILTIN_FIELDS else chunk.get("metadata", {}).get(key)) == value
                    for key, value in filters.items()
                )
            ]
        return [{**chunk, "strategy": "tfidf"} for chunk in results[:top_k]]


class HybridRetriever:
    """Reciprocal-rank fusion; component scores are not added or compared."""

    def __init__(self, dense: DenseRetriever, lexical: LexicalRetriever, *, rank_constant: int = 60):
        if rank_constant < 1:
            raise ValueError("rank_constant phải lớn hơn 0")
        self.dense = dense
        self.lexical = lexical
        self.rank_constant = rank_constant

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn 0")
        candidate_count = max(top_k * 3, top_k)
        ranked = (
            self.dense.search(query, filters=filters, top_k=candidate_count),
            self.lexical.search(query, filters=filters, top_k=candidate_count),
        )
        fused: dict[str, dict[str, Any]] = {}
        for results in ranked:
            for rank, chunk in enumerate(results, start=1):
                entry = fused.setdefault(chunk["id"], {**chunk, "rrf_score": 0.0})
                entry["rrf_score"] += 1.0 / (self.rank_constant + rank)
        return [
            {**chunk, "score": chunk.pop("rrf_score"), "strategy": "hybrid_rrf"}
            for chunk in sorted(fused.values(), key=lambda item: (-item["rrf_score"], item["id"]))[:top_k]
        ]


def create_dense_retriever(chunks: list[dict[str, Any]], persist_dir: Path, *,
                           model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                           collection_name: str = "day7_chunks") -> DenseRetriever:
    """Upsert chunks into a persistent Chroma collection and return its retriever."""
    import chromadb

    embedder = SentenceTransformerEmbedder(model_name)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )
    if chunks:
        if collection.metadata.get("embedding_model") != model_name:
            raise ValueError("Chroma index embedding model does not match the requested model")
        chunk_ids = [chunk["id"] for chunk in chunks]
        existing = collection.get(include=["documents"])
        existing_ids = set(existing["ids"])
        existing_documents = dict(zip(existing["ids"], existing["documents"]))
        documents = [chunk["text"] for chunk in chunks]
        if (existing_ids != set(chunk_ids)
                or any(existing_documents.get(chunk["id"]) != chunk["text"] for chunk in chunks)):
            stale_ids = existing_ids - set(chunk_ids)
            if stale_ids:
                collection.delete(ids=sorted(stale_ids))
            collection.upsert(
                ids=chunk_ids,
                documents=documents,
                metadatas=[_chroma_metadata(chunk) for chunk in chunks],
                embeddings=embedder.encode(documents),
            )
    return DenseRetriever(collection, embedder.encode)
