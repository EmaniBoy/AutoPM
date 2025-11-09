import os
import json
import numpy as np
from typing import List, Dict
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

class LocalVectorDB:
    """
    Lightweight local vector database that uses NVIDIA NIM embeddings.
    Stores embeddings and metadata in a JSON file for quick retrieval.
    """

    def __init__(self, api_key: str, embed_model: str, db_path: str):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model = embed_model
        self.db_path = db_path
        self.vectors: List[List[float]] = []
        self.meta: List[Dict] = []

        if os.path.exists(db_path):
            try:
                with open(db_path, "r") as f:
                    data = json.load(f)
                # Keep as lists so .append works - ensure they're actually lists
                vectors_data = data.get("vectors", []) or []
                meta_data = data.get("meta", []) or []
                # Ensure vectors are lists of lists (not numpy arrays)
                self.vectors = []
                for vec in vectors_data:
                    if isinstance(vec, list):
                        self.vectors.append(vec)
                    else:
                        self.vectors.append(list(vec))
                self.meta = meta_data if isinstance(meta_data, list) else []
                print(f"✅ Loaded {len(self.vectors)} vectors from {db_path}")
            except Exception as e:
                print(f"⚠️  Could not load vector DB: {e}")

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def add_document(self, doc_id: str, chunks: List[str]):
        count_before = len(self.vectors)
        for i, chunk in enumerate(chunks):
            emb = self.embed(chunk)
            # ensure emb is a list (not np.ndarray)
            if not isinstance(emb, list):
                emb = list(emb)
            self.vectors.append(emb)
            self.meta.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "text": chunk
            })
        self._save()
        print(f"📦 Stored {len(self.vectors) - count_before} chunks for {doc_id}")

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {"vectors": self.vectors, "meta": self.meta}  # lists serialize to JSON cleanly
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved {len(self.vectors)} vectors to {self.db_path}")

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        if not self.vectors:
            raise ValueError("❌ Vector DB is empty. Please run build_rag_index.py first to index documents.")
        try:
            query_vec = np.array(self.embed(text)).reshape(1, -1)
            matrix    = np.array(self.vectors, dtype=float)
            sims      = cosine_similarity(query_vec, matrix)[0]
            idxs      = np.argsort(sims)[::-1][:top_k]
            return [self.meta[i] for i in idxs]
        except Exception as e:
            raise ValueError(f"Error querying vector database: {str(e)}")
