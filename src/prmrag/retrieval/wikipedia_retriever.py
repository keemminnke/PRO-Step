"""Wikipedia-based retriever."""

from typing import List, Dict, Any
import wikipediaapi
import re


class WikipediaRetriever:
    """Wikipedia API-based retriever.

    Perfect for HotpotQA since it's Wikipedia-based.
    No API key needed, free to use.
    """

    def __init__(
        self,
        lang: str = 'en',
        user_agent: str = 'PRMRAG/1.0 (https://github.com/yourrepo)',
        extract_sentences: int = 5,
    ):
        """Initialize Wikipedia retriever.

        Args:
            lang: Language code (en, ko, etc.)
            user_agent: User agent for API requests
            extract_sentences: Number of sentences to extract per page
        """
        self.lang = lang
        self.extract_sentences = extract_sentences

        # Initialize Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            language=lang,
            user_agent=user_agent,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

        print(f"Wikipedia retriever initialized (lang={lang})")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k Wikipedia pages for query.

        Args:
            query: Search query
            top_k: Number of pages to retrieve

        Returns:
            List of dicts with keys: doc_id, title, text, url, score
        """
        # Search Wikipedia
        try:
            search_results = self._search_wikipedia(query, max_results=top_k * 2)
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []

        documents = []

        for i, title in enumerate(search_results[:top_k]):
            try:
                page = self.wiki.page(title)

                if not page.exists():
                    continue

                # Extract text (summary or first N sentences)
                text = self._extract_text(page)

                # Compute simple relevance score (position-based)
                score = 1.0 - (i * 0.1)

                documents.append({
                    'doc_id': str(page.pageid),
                    'title': page.title,
                    'text': text,
                    'url': page.fullurl,
                    'score': max(score, 0.1),  # Minimum 0.1
                })

            except Exception as e:
                print(f"Error fetching page '{title}': {e}")
                continue

        return documents

    def _search_wikipedia(self, query: str, max_results: int = 10) -> List[str]:
        """Search Wikipedia and return page titles.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of page titles
        """
        # Wikipedia API doesn't have built-in search in wikipediaapi
        # We need to use MediaWiki API directly
        import requests

        url = f"https://{self.lang}.wikipedia.org/w/api.php"
        params = {
            'action': 'opensearch',
            'search': query,
            'limit': max_results,
            'namespace': 0,  # Main namespace only
            'format': 'json',
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # OpenSearch returns [query, [titles], [descriptions], [urls]]
            if len(data) >= 2:
                return data[1]  # List of titles
            else:
                return []

        except Exception as e:
            print(f"Wikipedia search API error: {e}")
            return []

    def _extract_text(self, page: wikipediaapi.WikipediaPage) -> str:
        """Extract text from Wikipedia page.

        Args:
            page: Wikipedia page object

        Returns:
            Extracted text (summary or first N sentences)
        """
        # Get summary (first paragraph)
        summary = page.summary

        if summary:
            # Extract first N sentences
            sentences = self._split_sentences(summary)
            text = ' '.join(sentences[:self.extract_sentences])
            return text

        # Fallback: use full text (first 500 chars)
        return page.text[:500]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def retrieve_by_title(self, title: str) -> Dict[str, Any]:
        """Retrieve a specific Wikipedia page by title.

        Args:
            title: Wikipedia page title

        Returns:
            Document dict or None if not found
        """
        try:
            page = self.wiki.page(title)

            if not page.exists():
                return None

            text = self._extract_text(page)

            return {
                'doc_id': str(page.pageid),
                'title': page.title,
                'text': text,
                'url': page.fullurl,
                'score': 1.0,
            }

        except Exception as e:
            print(f"Error fetching page '{title}': {e}")
            return None

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """Retrieve for multiple queries.

        Args:
            queries: List of search queries
            top_k: Number of documents per query

        Returns:
            List of retrieval results
        """
        return [self.retrieve(q, top_k) for q in queries]


def create_retriever(retriever_type: str, **kwargs):
    """Factory function to create retrievers.

    Args:
        retriever_type: "bm25" or "wikipedia"
        **kwargs: Arguments for the retriever

    Returns:
        Retriever instance
    """
    if retriever_type == "bm25":
        from . import BM25Retriever
        return BM25Retriever(**kwargs)
    elif retriever_type == "wikipedia":
        return WikipediaRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
