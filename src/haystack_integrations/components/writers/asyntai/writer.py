"""Components that add knowledge to an Asyntai website agent.

Put ``AsyntaiDocumentWriter`` at the end of an indexing pipeline and every
document it receives becomes part of what the website chat agent knows.
``AsyntaiUrlIndexer`` asks Asyntai to read web pages itself.
"""

import os
from typing import Any, Dict, List, Optional

from haystack import Document, component, logging
from haystack.utils import Secret

from haystack_integrations.common.asyntai.base import _AsyntaiComponent
from haystack_integrations.common.asyntai.client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, AsyntaiError

logger = logging.getLogger(__name__)

# The Asyntai API refuses a text shorter than this.
MIN_CONTENT_CHARS = 10
MAX_TITLE_CHARS = 200


def document_title(document: Document) -> str:
    """The title Asyntai shows for a document in its knowledge base."""
    meta = document.meta or {}
    for key in ("title", "name"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:MAX_TITLE_CHARS]
    for key in ("file_path", "file_name", "url", "source"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            base = os.path.basename(value.strip().rstrip("/\\")) or value.strip()
            return base[:MAX_TITLE_CHARS]
    return f"Document {document.id[:12]}"


@component
class AsyntaiDocumentWriter(_AsyntaiComponent):
    """Stores Haystack documents in the knowledge base of an Asyntai agent.

    Usage:

    ```python
    from haystack import Pipeline
    from haystack.components.converters import TextFileToDocument
    from haystack_integrations.components.writers.asyntai import AsyntaiDocumentWriter

    pipeline = Pipeline()
    pipeline.add_component("converter", TextFileToDocument())
    pipeline.add_component("writer", AsyntaiDocumentWriter())
    pipeline.connect("converter.documents", "writer.documents")
    pipeline.run({"converter": {"sources": ["faq.txt", "returns.txt"]}})
    ```

    The title comes from ``meta["title"]``, then the file name, then the
    document id. Documents with less than 10 characters of text are skipped,
    because Asyntai refuses them.
    """

    def __init__(
        self,
        api_key: Secret = None,
        website_id: Optional[int] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        raise_on_failure: bool = True,
    ) -> None:
        """
        :param api_key: The Asyntai API key. Defaults to the
            ``ASYNTAI_API_KEY`` environment variable.
        :param website_id: The website agent to teach. Leave empty for the
            first website on the account.
        :param base_url: The Asyntai server.
        :param timeout: Seconds to wait for each upload.
        :param raise_on_failure: When ``False``, a refused document is logged
            and listed in ``failed``, and the run continues.
        """
        _AsyntaiComponent.__init__(self, api_key=api_key, website_id=website_id, base_url=base_url, timeout=timeout)
        self.raise_on_failure = raise_on_failure

    def _init_params(self) -> Dict[str, Any]:
        params = _AsyntaiComponent._init_params(self)
        params["raise_on_failure"] = self.raise_on_failure
        return params

    @component.output_types(documents_written=int, skipped=List[str], failed=List[Dict[str, str]])
    def run(self, documents: List[Document]):
        """
        :param documents: The documents to store.
        :returns: ``documents_written``, the ids of ``skipped`` documents
            (no or too little text), and ``failed`` uploads with the error.
        """
        written = 0
        skipped: List[str] = []
        failed: List[Dict[str, str]] = []
        for document in documents:
            content = (document.content or "").strip()
            if len(content) < MIN_CONTENT_CHARS:
                skipped.append(document.id)
                continue
            title = document_title(document)
            try:
                self.client.add_text(title=title, content=content, website_id=self.website_id)
            except AsyntaiError as exc:
                if self.raise_on_failure:
                    raise
                logger.warning(
                    "Asyntai refused document {doc_id}: {error}", doc_id=document.id, error=str(exc)
                )
                failed.append({"id": document.id, "title": title, "error": str(exc)})
                continue
            written += 1
        return {"documents_written": written, "skipped": skipped, "failed": failed}


@component
class AsyntaiUrlIndexer(_AsyntaiComponent):
    """Asks an Asyntai agent to read web pages into its knowledge base.

    Usage:

    ```python
    from haystack_integrations.components.writers.asyntai import AsyntaiUrlIndexer

    indexer = AsyntaiUrlIndexer()
    indexer.run(urls=["https://example.com/shipping", "https://example.com/returns"])
    ```
    """

    def __init__(
        self,
        api_key: Secret = None,
        website_id: Optional[int] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        raise_on_failure: bool = True,
    ) -> None:
        """
        :param api_key: The Asyntai API key. Defaults to the
            ``ASYNTAI_API_KEY`` environment variable.
        :param website_id: The website agent to teach. Leave empty for the
            first website on the account.
        :param base_url: The Asyntai server.
        :param timeout: Seconds to wait for each page.
        :param raise_on_failure: When ``False``, a refused URL is logged and
            listed in ``failed``, and the run continues.
        """
        _AsyntaiComponent.__init__(self, api_key=api_key, website_id=website_id, base_url=base_url, timeout=timeout)
        self.raise_on_failure = raise_on_failure

    def _init_params(self) -> Dict[str, Any]:
        params = _AsyntaiComponent._init_params(self)
        params["raise_on_failure"] = self.raise_on_failure
        return params

    @component.output_types(indexed=List[Dict[str, Any]], failed=List[Dict[str, str]])
    def run(self, urls: List[str]):
        """
        :param urls: The pages to read.
        :returns: ``indexed`` pages with their title and chunk count, and
            ``failed`` pages with the error.
        """
        indexed: List[Dict[str, Any]] = []
        failed: List[Dict[str, str]] = []
        for url in urls:
            try:
                payload = self.client.add_url(url=url, website_id=self.website_id)
            except AsyntaiError as exc:
                if self.raise_on_failure:
                    raise
                logger.warning("Asyntai could not read {url}: {error}", url=url, error=str(exc))
                failed.append({"url": url, "error": str(exc)})
                continue
            indexed.append({
                "url": url,
                "title": payload.get("title", url),
                "chunks": payload.get("chunks_created", 0),
            })
        return {"indexed": indexed, "failed": failed}
