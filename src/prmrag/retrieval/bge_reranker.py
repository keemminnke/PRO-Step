"""BGE Reranker for improving retrieval quality.

Uses BAAI/bge-reranker-v2-m3 to rerank retrieved documents based on
query-document relevance scores.
"""

from typing import List, Dict, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReranker:
    """BGE Reranker using BAAI/bge-reranker-v2-m3.

    The reranker computes relevance scores between query and documents,
    improving ranking quality over pure BM25/BGE retrieval.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """Initialize BGE reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
            batch_size: Batch size for reranking
        """
        self.device = device
        self.batch_size = batch_size

        print(f"Loading BGE Reranker: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ BGE Reranker loaded on {device}")

    def _compute_scores(
        self,
        query: str,
        texts: List[str]
    ) -> List[float]:
        """Compute relevance scores for query-text pairs.

        Args:
            query: Search query
            texts: List of document texts

        Returns:
            List of relevance scores (higher = more relevant)
        """
        scores = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Create query-document pairs
            pairs = [[query, text] for text in batch_texts]

            # Tokenize
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # Compute scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().tolist()

                # Handle single item case
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]

                scores.extend(batch_scores)

        return scores

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query-document relevance.

        Args:
            query: Search query
            documents: List of documents (must have 'text' field)
            top_k: Number of top documents to return (None = all)

        Returns:
            Reranked documents with 'rerank_score' field
        """
        if not documents:
            return []

        # Extract texts
        texts = [doc['text'] for doc in documents]

        # Compute reranking scores
        rerank_scores = self._compute_scores(query, texts)

        # Add scores to documents
        reranked_docs = []
        for doc, score in zip(documents, rerank_scores):
            reranked_doc = doc.copy()
            reranked_doc['rerank_score'] = score
            reranked_docs.append(reranked_doc)

        # Sort by rerank score (descending)
        reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Return top-k
        if top_k is not None:
            return reranked_docs[:top_k]
        return reranked_docs

    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]],
        top_k: int = None
    ) -> List[List[Dict[str, Any]]]:
        """Batch rerank documents for multiple queries.

        Processes all query-document pairs in batches for GPU efficiency.

        Args:
            queries: List of search queries
            documents_list: List of document lists (one per query)
            top_k: Number of top documents to return per query

        Returns:
            List of reranked document lists
        """
        if not queries or not documents_list:
            return []

        # Flatten all query-document pairs with tracking indices
        all_pairs = []
        pair_indices = []  # (query_idx, doc_idx)

        for query_idx, (query, documents) in enumerate(zip(queries, documents_list)):
            for doc_idx, doc in enumerate(documents):
                all_pairs.append([query, doc['text']])
                pair_indices.append((query_idx, doc_idx))

        if not all_pairs:
            return [[] for _ in queries]

        # Compute all scores in batches
        all_scores = []
        for i in range(0, len(all_pairs), self.batch_size):
            batch_pairs = all_pairs[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # Compute scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().tolist()

                # Handle single item case
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]

                all_scores.extend(batch_scores)

        # Distribute scores back to queries
        results = [[] for _ in queries]
        for (query_idx, doc_idx), score in zip(pair_indices, all_scores):
            doc = documents_list[query_idx][doc_idx].copy()
            doc['rerank_score'] = score
            results[query_idx].append(doc)

        # Sort each query's results and apply top_k
        for i in range(len(results)):
            results[i].sort(key=lambda x: x['rerank_score'], reverse=True)
            if top_k is not None:
                results[i] = results[i][:top_k]

        return results
