"""arXiv research paper scraper for industry-adjacent topics."""

from __future__ import annotations

import logging

import arxiv

from src.ingestion.base import BaseScraper, Document, make_document_id


logger = logging.getLogger(__name__)


# Topics relevant to BMW's strategic context; tracked to surface
# emerging research trends in EVs, autonomy, batteries, and supply chain.
DEFAULT_ARXIV_QUERIES = [
    "electric vehicle battery",
    "autonomous driving perception",
    "lithium-ion battery management",
    "automotive AI",
    "vehicle-to-grid",
]


class ArxivScraper(BaseScraper):
    """Pull recent papers from arXiv on topics relevant to BMW."""

    name = "arxiv"

    def __init__(
        self,
        queries: list[str] | None = None,
        max_results_per_query: int = 10,
        page_size: int = 20,
        delay_seconds: float = 5.0,
        num_retries: int = 3,
    ) -> None:
        self.queries = queries or DEFAULT_ARXIV_QUERIES
        self.max_results_per_query = max_results_per_query
        self._client = arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries,
        )

    def fetch(self) -> list[Document]:
        documents: dict[str, Document] = {}

        for query in self.queries:
            search = arxiv.Search(
                query=query,
                max_results=self.max_results_per_query,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            try:
                results = list(self._client.results(search))
            except Exception as exc:
                logger.warning("arXiv query %r failed: %s", query, exc)
                continue

            for result in results:
                doc = self._result_to_document(result, query)
                if doc.id not in documents:
                    documents[doc.id] = doc

        logger.info("arXiv: %d unique documents", len(documents))
        return list(documents.values())

    @classmethod
    def _result_to_document(cls, result, query: str) -> Document:
        url = result.entry_id
        title = result.title.strip()
        abstract = result.summary.strip().replace("\n", " ")
        text = f"{title}\n\n{abstract}"

        return Document(
            id=make_document_id(cls.name, url),
            source=cls.name,
            url=url,
            title=title,
            text=text,
            published_at=result.published,
            metadata={
                "query": query,
                "authors": [a.name for a in result.authors],
                "categories": result.categories,
                "primary_category": result.primary_category,
                "pdf_url": result.pdf_url,
            },
        )
