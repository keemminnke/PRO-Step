"""BGE-M3 dense retriever with FAISS-GPU for fast similarity search."""

import warnings
# Suppress XLMRobertaTokenizerFast warning from BGE-M3
warnings.filterwarnings("ignore", message="You're using a XLMRobertaTokenizerFast")

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import pickle

# Import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-gpu")


class BGERetriever:
    """BGE-M3 dense retriever with FAISS-GPU acceleration."""

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
        embedding_cache_path: Optional[str] = None,
        use_gpu: bool = False,  # CPU FAISS (GPU FAISS uses too much memory with large corpus)
        faiss_batch_size: int = 256,  # Batch size for FAISS search
        faiss_index_path: Optional[str] = None,  # Path to save/load FAISS index
    ):
        """Initialize BGE retriever with FAISS-GPU.

        Args:
            corpus: List of documents, each with 'id', 'title', 'text'
            model_name: HuggingFace model name for BGE
            batch_size: Batch size for encoding
            max_length: Max sequence length
            device: Device to use (None = auto-detect)
            embedding_cache_path: Path to load/save corpus embeddings
            use_gpu: Whether to use GPU for FAISS (default: True)
        """
        self.corpus = corpus
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name
        self.embedding_cache_path = embedding_cache_path
        self.use_gpu = use_gpu and FAISS_AVAILABLE and torch.cuda.is_available()
        self.faiss_batch_size = faiss_batch_size  # Batch size for FAISS search
        self.faiss_index_path = faiss_index_path

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading BGE-M3 model on {self.device}...")
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=(self.device == "cuda"),
            devices=self.device,
        )
        print("✓ BGE-M3 model loaded")

        # Try loading saved FAISS index first (fastest path)
        if faiss_index_path and Path(faiss_index_path).exists() and FAISS_AVAILABLE:
            print(f"Loading saved FAISS index from {faiss_index_path}...")
            self.faiss_index = faiss.read_index(faiss_index_path)
            self.doc_embeddings = None
            print(f"✓ FAISS index loaded ({self.faiss_index.ntotal} vectors)")
        else:
            # Load or build embeddings
            if embedding_cache_path and Path(embedding_cache_path).exists():
                print(f"Loading cached embeddings from {embedding_cache_path}...")
                self.doc_embeddings = self._load_embeddings(embedding_cache_path)
                print(f"✓ Loaded {len(self.doc_embeddings)} cached embeddings")
            else:
                print(f"Building dense embeddings for {len(corpus)} documents...")
                self.doc_embeddings = self._encode_corpus()
                print("✓ Dense embeddings built!")

                if embedding_cache_path:
                    self._save_embeddings(embedding_cache_path)
                    print(f"✓ Saved embeddings to {embedding_cache_path}")

            # Build FAISS index
            self._build_faiss_index()

            # Save FAISS index to disk for fast reloading next time
            if faiss_index_path and self.faiss_index is not None:
                faiss.write_index(self.faiss_index, faiss_index_path)
                print(f"✓ Saved FAISS index to {faiss_index_path}")

            # Free raw embeddings after FAISS index is built (saves ~24GB RAM)
            if self.faiss_index is not None:
                del self.doc_embeddings
                self.doc_embeddings = None
                import gc; gc.collect()
                print("  ✓ Freed raw embeddings (FAISS index holds normalized copy)")

    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        if not FAISS_AVAILABLE:
            print("Warning: FAISS not available, falling back to numpy (slow)")
            self.faiss_index = None
            return

        print("Building FAISS index...")

        # Get embedding dimension
        embed_dim = self.doc_embeddings.shape[1]

        # Normalize embeddings for cosine similarity (inner product on normalized = cosine)
        embeddings = self.doc_embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        # Create index (IndexFlatIP = Inner Product, use with normalized vectors for cosine sim)
        self.faiss_index = faiss.IndexFlatIP(embed_dim)

        if self.use_gpu:
            print("  Moving FAISS index to GPU...")
            # Use GPU 0
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
            print("  ✓ FAISS index on GPU")
        else:
            print("  Using CPU FAISS index")

        # Add embeddings to index
        self.faiss_index.add(embeddings)
        print(f"✓ FAISS index built with {self.faiss_index.ntotal} vectors")

    def _encode_corpus(self) -> np.ndarray:
        """Encode all corpus documents into embeddings."""
        texts = [f"{doc['title']} {doc['text']}" for doc in self.corpus]

        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding corpus"):
            batch = texts[i:i + self.batch_size]
            embeddings = self.model.encode(
                batch,
                max_length=self.max_length,
                batch_size=len(batch),
            )['dense_vecs']
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def _save_embeddings(self, cache_path: str):
        """Save corpus embeddings to disk."""
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, self.doc_embeddings)

    def _load_embeddings(self, cache_path: str) -> np.ndarray:
        """Load corpus embeddings from disk."""
        try:
            embeddings = np.load(cache_path)
            if len(embeddings) > len(self.corpus):
                print(f"  Warning: Embedding file has {len(embeddings)} docs, "
                      f"but corpus has {len(self.corpus)} docs. Using first {len(self.corpus)} embeddings.")
                return embeddings[:len(self.corpus)]
            return embeddings
        except (ValueError, pickle.UnpicklingError):
            # Fall back to memmap loading
            file_size = Path(cache_path).stat().st_size
            embed_dim = 1024
            bytes_per_float16 = 2
            num_docs_in_file = file_size // (embed_dim * bytes_per_float16)

            full_embeddings = np.memmap(
                cache_path,
                dtype=np.float16,
                mode='r',
                shape=(num_docs_in_file, embed_dim)
            )

            if num_docs_in_file > len(self.corpus):
                print(f"  Warning: Embedding file has {num_docs_in_file} docs, "
                      f"but corpus has {len(self.corpus)} docs. Using first {len(self.corpus)} embeddings.")
                return np.array(full_embeddings[:len(self.corpus)])
            elif num_docs_in_file < len(self.corpus):
                raise ValueError(
                    f"Embedding file has only {num_docs_in_file} docs, "
                    f"but corpus has {len(self.corpus)} docs"
                )

            return np.array(full_embeddings)  # Load into memory for FAISS

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for a query using FAISS."""
        # Encode query
        query_embedding = self.model.encode(
            [query],
            max_length=self.max_length,
            batch_size=1,
        )['dense_vecs'][0]

        # Normalize for cosine similarity
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        if self.faiss_index is not None:
            # FAISS search (GPU accelerated)
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1), top_k
            )
            scores = scores[0]
            indices = indices[0]
        else:
            # Fallback to numpy (slow)
            doc_norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
            doc_embeddings_norm = self.doc_embeddings / doc_norms
            scores = np.dot(doc_embeddings_norm, query_embedding)
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]

        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:  # FAISS returns -1 for empty slots
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
        """Batch retrieve for multiple queries using FAISS."""
        if not queries:
            return []

        # Encode all queries at once
        query_embeddings = self.model.encode(
            queries,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )['dense_vecs']

        # Normalize for cosine similarity
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)

        if self.faiss_index is not None:
            # FAISS batch search (GPU accelerated) - process in smaller batches to avoid CUBLAS errors
            num_queries = len(query_embeddings)
            if num_queries <= self.faiss_batch_size:
                scores_matrix, indices_matrix = self.faiss_index.search(query_embeddings, top_k)
            else:
                # Process in batches
                all_scores = []
                all_indices = []
                for i in range(0, num_queries, self.faiss_batch_size):
                    batch = query_embeddings[i:i + self.faiss_batch_size]
                    scores, indices = self.faiss_index.search(batch, top_k)
                    all_scores.append(scores)
                    all_indices.append(indices)
                scores_matrix = np.vstack(all_scores)
                indices_matrix = np.vstack(all_indices)
        else:
            # Fallback to numpy (slow)
            doc_norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
            doc_embeddings_norm = self.doc_embeddings / doc_norms
            query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            query_embeddings_norm = query_embeddings / query_norms
            scores_all = np.dot(query_embeddings_norm, doc_embeddings_norm.T)
            indices_matrix = np.argsort(scores_all, axis=1)[:, ::-1][:, :top_k]
            scores_matrix = np.take_along_axis(scores_all, indices_matrix, axis=1)

        # Build results
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
