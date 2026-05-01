"""Hybrid retriever combining BM25 (sparse) and BGE (dense) with rank fusion."""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict


class HybridRetriever:
    """Hybrid retriever using BM25 + BGE with rank fusion.

    Supports two fusion methods:
    1. RRF (Reciprocal Rank Fusion) - recommended, no normalization needed
    2. Weighted sum - requires score normalization
    """

    def __init__(
        self,
        bm25_retriever,
        bge_retriever,
        fusion_method: str = "rrf",  # "rrf" or "weighted"
        k_sparse: int = 50,  # Top-K for BM25
        k_dense: int = 50,   # Top-K for BGE
        alpha: float = 0.5,  # Weight for weighted sum (alpha * BM25 + (1-alpha) * BGE)
        rrf_k: int = 60,     # Constant for RRF formula
        reranker = None,     # Optional BGE reranker
        rerank_top_n: int = 20,  # Rerank top-N candidates from fusion
    ):
        """Initialize hybrid retriever.

        Args:
            bm25_retriever: BM25 retriever instance
            bge_retriever: BGE retriever instance
            fusion_method: "rrf" or "weighted"
            k_sparse: Top-K documents to retrieve from BM25
            k_dense: Top-K documents to retrieve from BGE
            alpha: Weight for BM25 in weighted sum (0.5 = equal weight)
            rrf_k: Constant for RRF formula (typically 60)
        """
        self.bm25 = bm25_retriever
        self.bge = bge_retriever
        self.fusion_method = fusion_method
        self.k_sparse = k_sparse
        self.k_dense = k_dense
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n

        print(f"✓ Hybrid retriever initialized:")
        print(f"  - Fusion method: {fusion_method}")
        print(f"  - BM25 top-K: {k_sparse}")
        print(f"  - BGE top-K: {k_dense}")
        if fusion_method == "weighted":
            print(f"  - Alpha (BM25 weight): {alpha}")
        elif fusion_method == "rrf":
            print(f"  - RRF constant k: {rrf_k}")
        if reranker is not None:
            print(f"  - Reranker: enabled (rerank top-{rerank_top_n} candidates)")

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalize scores to [0, 1].

        Args:
            scores: List of scores

        Returns:
            Normalized scores
        """
        if not scores or len(scores) == 1:
            return [1.0] * len(scores)

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _fuse_rrf(
        self,
        bm25_results: List[Dict[str, Any]],
        bge_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Fuse results using Reciprocal Rank Fusion (RRF).

        RRF formula: score(d) = sum_r [ 1 / (k + rank_r(d)) ]
        where r iterates over retrievers (BM25, BGE)

        Args:
            bm25_results: Results from BM25
            bge_results: Results from BGE

        Returns:
            Dict mapping doc_id to fused score
        """
        fused_scores = defaultdict(float)

        # Add BM25 ranks
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result['doc_id']
            fused_scores[doc_id] += 1.0 / (self.rrf_k + rank)

        # Add BGE ranks
        for rank, result in enumerate(bge_results, start=1):
            doc_id = result['doc_id']
            fused_scores[doc_id] += 1.0 / (self.rrf_k + rank)

        return fused_scores

    def _fuse_weighted(
        self,
        bm25_results: List[Dict[str, Any]],
        bge_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Fuse results using weighted sum of normalized scores.

        Final score = alpha * norm(BM25) + (1 - alpha) * norm(BGE)

        Args:
            bm25_results: Results from BM25
            bge_results: Results from BGE

        Returns:
            Dict mapping doc_id to fused score
        """
        # Normalize BM25 scores
        bm25_scores = [r['score'] for r in bm25_results]
        bm25_scores_norm = self._normalize_scores(bm25_scores)

        bm25_score_map = {
            result['doc_id']: score_norm
            for result, score_norm in zip(bm25_results, bm25_scores_norm)
        }

        # Normalize BGE scores
        bge_scores = [r['score'] for r in bge_results]
        bge_scores_norm = self._normalize_scores(bge_scores)

        bge_score_map = {
            result['doc_id']: score_norm
            for result, score_norm in zip(bge_results, bge_scores_norm)
        }

        # Combine scores with weighted sum
        all_doc_ids = set(bm25_score_map.keys()) | set(bge_score_map.keys())

        fused_scores = {}
        for doc_id in all_doc_ids:
            score_bm25 = bm25_score_map.get(doc_id, 0.0)
            score_bge = bge_score_map.get(doc_id, 0.0)
            fused_scores[doc_id] = self.alpha * score_bm25 + (1 - self.alpha) * score_bge

        return fused_scores

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for a query using hybrid retrieval.

        Args:
            query: Search query
            top_k: Number of final documents to return

        Returns:
            List of retrieved documents with fused scores (and rerank scores if reranker enabled)
        """
        # Step 1: Retrieve from both BM25 and BGE
        bm25_results = self.bm25.retrieve(query, top_k=self.k_sparse)
        bge_results = self.bge.retrieve(query, top_k=self.k_dense)

        # Step 2: Fuse scores
        if self.fusion_method == "rrf":
            fused_scores = self._fuse_rrf(bm25_results, bge_results)
        elif self.fusion_method == "weighted":
            fused_scores = self._fuse_weighted(bm25_results, bge_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Step 3: Get candidates for reranking (or final results if no reranker)
        # If reranker is enabled, get more candidates (rerank_top_n)
        # Otherwise, just get top_k
        num_candidates = self.rerank_top_n if self.reranker else top_k

        sorted_doc_ids = sorted(
            fused_scores.keys(),
            key=lambda doc_id: fused_scores[doc_id],
            reverse=True
        )[:num_candidates]

        # Step 4: Build candidate results
        # Create a doc_id -> doc mapping for quick lookup
        doc_map = {}
        for result in bm25_results + bge_results:
            doc_id = result['doc_id']
            if doc_id not in doc_map:
                doc_map[doc_id] = result

        candidate_results = []
        for doc_id in sorted_doc_ids:
            doc = doc_map[doc_id]
            candidate_results.append({
                'doc_id': doc_id,
                'title': doc['title'],
                'text': doc['text'],
                'score': fused_scores[doc_id],
            })

        # Step 5: Rerank if reranker is available
        if self.reranker:
            final_results = self.reranker.rerank(query, candidate_results, top_k=top_k)
        else:
            final_results = candidate_results

        return final_results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """Batch retrieve for multiple queries using efficient batching.

        This method batches the underlying BGE retriever calls for much faster
        embedding encoding (single batch vs. N individual calls).

        Args:
            queries: List of search queries
            top_k: Number of documents per query

        Returns:
            List of retrieval results
        """
        if not queries:
            return []

        # Batch BM25 retrieval (usually fast, but batch if supported)
        if hasattr(self.bm25, 'batch_retrieve'):
            all_bm25_results = self.bm25.batch_retrieve(queries, top_k=self.k_sparse)
        else:
            all_bm25_results = [self.bm25.retrieve(q, top_k=self.k_sparse) for q in queries]

        # Batch BGE retrieval (critical for performance - encodes all queries at once)
        if hasattr(self.bge, 'batch_retrieve'):
            all_bge_results = self.bge.batch_retrieve(queries, top_k=self.k_dense)
        else:
            all_bge_results = [self.bge.retrieve(q, top_k=self.k_dense) for q in queries]

        # Fuse and optionally rerank for each query
        all_candidates = []
        for i, query in enumerate(queries):
            bm25_results = all_bm25_results[i]
            bge_results = all_bge_results[i]

            # Fuse scores
            if self.fusion_method == "rrf":
                fused_scores = self._fuse_rrf(bm25_results, bge_results)
            elif self.fusion_method == "weighted":
                fused_scores = self._fuse_weighted(bm25_results, bge_results)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

            # Get candidates
            num_candidates = self.rerank_top_n if self.reranker else top_k
            sorted_doc_ids = sorted(
                fused_scores.keys(),
                key=lambda doc_id: fused_scores[doc_id],
                reverse=True
            )[:num_candidates]

            # Build candidate results
            doc_map = {}
            for result in bm25_results + bge_results:
                doc_id = result['doc_id']
                if doc_id not in doc_map:
                    doc_map[doc_id] = result

            candidate_results = []
            for doc_id in sorted_doc_ids:
                doc = doc_map[doc_id]
                candidate_results.append({
                    'doc_id': doc_id,
                    'title': doc['title'],
                    'text': doc['text'],
                    'score': fused_scores[doc_id],
                })

            all_candidates.append(candidate_results)

        # Batch rerank all candidates at once (much faster than sequential)
        if self.reranker:
            if hasattr(self.reranker, 'batch_rerank'):
                all_results = self.reranker.batch_rerank(queries, all_candidates, top_k=top_k)
            else:
                # Fallback to sequential reranking
                all_results = []
                for query, candidates in zip(queries, all_candidates):
                    reranked = self.reranker.rerank(query, candidates, top_k=top_k)
                    all_results.append(reranked)
        else:
            all_results = all_candidates

        return all_results
