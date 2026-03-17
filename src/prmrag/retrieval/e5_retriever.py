"""E5 dense retriever using intfloat/e5-large-v2 with FAISS for similarity search."""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-gpu")


def _average_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over non-padding tokens."""
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5Retriever:
    """E5 dense retriever using intfloat/e5-large-v2 with FAISS.

    E5 requires prefix:
      - "query: <text>"  for queries
      - "passage: <text>" for documents
    """

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        model_name: str = "intfloat/e5-large-v2",
        batch_size: int = 64,
        max_length: int = 512,
        device: Optional[str] = None,
        embedding_cache_path: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
    ):
        self.corpus = corpus
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name
        self.embedding_cache_path = embedding_cache_path
        self.faiss_index_path = faiss_index_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        print(f"Loading E5 model ({model_name}) on {self.device} ({self.num_gpus} GPUs)...")
        from transformers import AutoTokenizer, AutoModel
        HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
        self._base_model = AutoModel.from_pretrained(
            model_name, cache_dir=HF_CACHE_DIR, torch_dtype=torch.float16
        ).to(self.device)
        self._base_model.eval()

        # Use DataParallel if multiple GPUs available
        if self.num_gpus >= 2:
            self.model = torch.nn.DataParallel(self._base_model)
            print(f"✓ E5 model loaded (DataParallel on {self.num_gpus} GPUs)")
        else:
            self.model = self._base_model
            print(f"✓ E5 model loaded")

        # Load or build index
        if faiss_index_path and Path(faiss_index_path).exists() and FAISS_AVAILABLE:
            print(f"Loading saved FAISS index from {faiss_index_path}...")
            self.faiss_index = faiss.read_index(faiss_index_path)
            print(f"✓ FAISS index loaded ({self.faiss_index.ntotal} vectors)")
        else:
            # Load or build embeddings
            if embedding_cache_path and Path(embedding_cache_path).exists():
                print(f"Loading cached embeddings from {embedding_cache_path}...")
                self.doc_embeddings = np.load(embedding_cache_path)
                print(f"✓ Loaded {len(self.doc_embeddings)} cached embeddings")
            else:
                print(f"Building E5 embeddings for {len(corpus)} documents...")
                self.doc_embeddings = self._encode_corpus()
                if embedding_cache_path:
                    Path(embedding_cache_path).parent.mkdir(parents=True, exist_ok=True)
                    np.save(embedding_cache_path, self.doc_embeddings)
                    print(f"✓ Saved embeddings to {embedding_cache_path}")

            self._build_faiss_index()

            if faiss_index_path and self.faiss_index is not None:
                Path(faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.faiss_index, str(faiss_index_path))
                print(f"✓ Saved FAISS index to {faiss_index_path}")

            del self.doc_embeddings
            self.doc_embeddings = None
            import gc; gc.collect()

        # Free E5 model from GPU after index is ready (frees memory for vLLM)
        self._free_model()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts with mean pooling + L2 normalization."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
            embeddings = _average_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.float().cpu().numpy())
        return np.vstack(all_embeddings)

    def _encode_corpus(self) -> np.ndarray:
        """Encode all corpus documents with 'passage: ' prefix."""
        texts = [f"passage: {doc['title']} {doc['text']}" for doc in self.corpus]
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding corpus (E5)"):
            batch = texts[i:i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
            embeddings = _average_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.float().cpu().numpy())
        return np.vstack(all_embeddings)

    def _free_model(self):
        """Free GPU memory after index is built."""
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self, "_base_model") and self._base_model is not None:
                self._base_model.cpu()
                del self._base_model
                self._base_model = None
            del self.model
            self.model = None
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ E5 model freed from GPU (FAISS index ready)")

    def _ensure_model(self):
        """Reload model on CPU for query-time encoding (avoids GPU conflict with vLLM)."""
        if self.model is None:
            print("Reloading E5 model on CPU for query encoding...")
            from transformers import AutoTokenizer, AutoModel
            HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            self.model = AutoModel.from_pretrained(
                self.model_name, cache_dir=HF_CACHE_DIR, torch_dtype=torch.float32
            ).cpu()
            self.model.eval()
            self.device = "cpu"
            print("✓ E5 model reloaded on CPU")

    def _build_faiss_index(self):
        if not FAISS_AVAILABLE:
            print("Warning: FAISS not available, falling back to numpy")
            self.faiss_index = None
            return
        print("Building FAISS index...")
        embed_dim = self.doc_embeddings.shape[1]
        embeddings = self.doc_embeddings.astype(np.float32)
        # Already normalized, use inner product (= cosine similarity)
        self.faiss_index = faiss.IndexFlatIP(embed_dim)
        self.faiss_index.add(embeddings)
        print(f"✓ FAISS index built with {self.faiss_index.ntotal} vectors")

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query with 'query: ' prefix."""
        text = f"query: {query}"
        encoded = self.tokenizer(
            [text],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded)
        embedding = _average_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.float().cpu().numpy()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for a query."""
        self._ensure_model()
        query_emb = self._encode_query(query)  # shape (1, D)
        scores, indices = self.faiss_index.search(query_emb, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            doc = self.corpus[idx]
            results.append({
                'doc_id': doc['id'],
                'title': doc['title'],
                'text': doc['text'],
                'score': float(score),
            })
        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """Batch retrieve for multiple queries."""
        if not queries:
            return []
        self._ensure_model()
        texts = [f"query: {q}" for q in queries]
        query_embeddings = self._encode_texts(texts)  # (N, D)
        scores_matrix, indices_matrix = self.faiss_index.search(query_embeddings, top_k)
        all_results = []
        for i in range(len(queries)):
            results = []
            for idx, score in zip(indices_matrix[i], scores_matrix[i]):
                if idx < 0:
                    continue
                doc = self.corpus[idx]
                results.append({
                    'doc_id': doc['id'],
                    'title': doc['title'],
                    'text': doc['text'],
                    'score': float(score),
                })
            all_results.append(results)
        return all_results
