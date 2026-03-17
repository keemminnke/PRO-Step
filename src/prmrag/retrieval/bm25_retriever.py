"""BM25 sparse retriever for lexical search."""

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from pathlib import Path
import pickle


class BM25Retriever:
    """BM25 sparse retriever using keyword matching."""

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        tokenizer: Optional[callable] = None,
        index_cache_path: Optional[str] = None,
    ):
        """Initialize BM25 retriever.

        Args:
            corpus: List of documents, each with 'id', 'title', 'text'
            tokenizer: Function to tokenize text (default: simple split)
            index_cache_path: Path to load/save BM25 index (None = don't cache)
        """
        self.corpus = corpus
        self.tokenizer = tokenizer or self._default_tokenizer
        self.index_cache_path = index_cache_path

        # Load or build BM25 index
        if index_cache_path and Path(index_cache_path).exists():
            print(f"Loading cached BM25 index from {index_cache_path}...")
            self.bm25 = self._load_index(index_cache_path)
            print(f"✓ Loaded BM25 index with {len(self.corpus)} documents")
        else:
            print(f"Building BM25 index for {len(corpus)} documents...")
            self.bm25 = self._build_index()
            print("✓ BM25 index built!")

            # Save index if cache path provided
            if index_cache_path:
                self._save_index(index_cache_path)
                print(f"✓ Saved BM25 index to {index_cache_path}")

    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenizer: lowercase and split on whitespace.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        return text.lower().split()

    def _build_index(self) -> BM25Okapi:
        """Build BM25 index from corpus.

        Returns:
            BM25Okapi index
        """
        # Tokenize all documents (title + text)
        tokenized_corpus = [
            self.tokenizer(f"{doc['title']} {doc['text']}")
            for doc in self.corpus
        ]

        # Build BM25 index
        return BM25Okapi(tokenized_corpus)

    def _save_index(self, cache_path: str):
        """Save BM25 index to disk.

        Args:
            cache_path: Path to save index
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, 'wb') as f:
            pickle.dump(self.bm25, f)

    def _load_index(self, cache_path: str) -> BM25Okapi:
        """Load BM25 index from disk.

        Args:
            cache_path: Path to load index from

        Returns:
            Loaded BM25Okapi index
        """
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        # Tokenize query
        tokenized_query = self.tokenizer(query)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_k_indices:
            doc = self.corpus[idx]
            results.append({
                'doc_id': doc['id'],
                'title': doc['title'],
                'text': doc['text'],
                'score': float(scores[idx]),
            })

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """Retrieve for multiple queries.

        Args:
            queries: List of search queries
            top_k: Number of documents per query

        Returns:
            List of retrieval results
        """
        all_results = []
        for query in queries:
            results = self.retrieve(query, top_k)
            all_results.append(results)

        return all_results
