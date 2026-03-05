"""
Weaviate RAG Client
Replaces ChromaDB for knowledge-base retrieval.

Config (env vars):
  WEAVIATE_RAG_URL   — base URL of the FN-Weaviate-DB API
                       e.g. http://46.62.157.117:8001/api/v1
  CHATBOT_API_KEY    — bot API key  (rag_xxx...)
  CHATBOT_ID         — bot UUID (informational, not sent in queries)
"""

import os
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

_WEAVIATE_RAG_URL = os.getenv("WEAVIATE_RAG_URL", "http://46.62.157.117:8001/api/v1")
_CHATBOT_API_KEY  = os.getenv("CHATBOT_API_KEY", "")
_CHATBOT_ID       = os.getenv("CHATBOT_ID", "")


class WeaviateRAGClient:
    """
    Thin HTTP wrapper around the FN-Weaviate-DB /query endpoint.

    Authentication: bot API key via  X-API-Key  header.
    Tenant + bot scoping is enforced server-side — callers need nothing else.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 15,
    ):
        self.base_url = (base_url or _WEAVIATE_RAG_URL).rstrip("/")
        self.api_key  = api_key or _CHATBOT_API_KEY
        self.timeout  = timeout

        if not self.api_key:
            logger.warning(
                "WeaviateRAGClient: CHATBOT_API_KEY not set — queries will fail auth."
            )

        self._headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        logger.info(
            f"WeaviateRAGClient ready  |  url={self.base_url}  "
            f"key_prefix={self.api_key[:12]}..."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        similarity_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Query the Weaviate RAG backend.

        Returns a dict in the same shape the old ChromaDB layer returned:
        {
            "success": bool,
            "results": [{"document": str, "chunk_id": str, "document_name": str,
                          "score": float, "source": str|None}, ...],
            "distances": [float, ...],      # 1 - score (kept for compat)
            "error": str | None
        }
        """
        payload = {
            "query": query_text,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "similarity_threshold": similarity_threshold,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/query",
                json=payload,
                headers=self._headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            distances = []

            for chunk in data.get("results", []):
                score = chunk.get("score", 0.0)
                results.append({
                    "document":      chunk.get("content", ""),
                    "chunk_id":      chunk.get("chunk_id", ""),
                    "document_name": chunk.get("document_name", ""),
                    "score":         score,
                    "source":        chunk.get("source"),
                    "bot_id":        chunk.get("bot_id"),
                    "collection_id": chunk.get("collection_id"),
                })
                # ChromaDB used "distance" (lower=better); Weaviate uses "score" (higher=better).
                # Convert so existing metrics logging stays meaningful.
                distances.append(round(1.0 - score, 4))

            logger.info(
                f"WeaviateRAG query OK  |  chunks={len(results)}  "
                f"query='{query_text[:60]}...'"
            )

            return {
                "success":   True,
                "results":   results,
                "distances": distances,
                "query_id":  data.get("query_id"),
                "search_time_ms": data.get("search_time_ms"),
                "error":     None,
            }

        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = str(exc)
            logger.error(f"WeaviateRAG HTTP error {exc.response.status_code}: {detail}")
            return {"success": False, "results": [], "distances": [], "error": detail}

        except Exception as exc:
            logger.error(f"WeaviateRAG request failed: {exc}")
            return {"success": False, "results": [], "distances": [], "error": str(exc)}

    def health_check(self) -> bool:
        """
        Ping the Weaviate server.
        Returns True if reachable, False otherwise.
        """
        try:
            resp = requests.get(
                f"{self.base_url.replace('/api/v1', '')}/health",
                timeout=5,
            )
            ok = resp.status_code == 200
            if ok:
                logger.info("WeaviateRAG health check: OK")
            else:
                logger.warning(f"WeaviateRAG health check: {resp.status_code}")
            return ok
        except Exception as exc:
            logger.warning(f"WeaviateRAG health check failed: {exc}")
            return False


# ---------------------------------------------------------------------------
# Module-level singleton so tools can import and reuse one instance
# ---------------------------------------------------------------------------
_client: Optional[WeaviateRAGClient] = None


def get_weaviate_rag_client() -> WeaviateRAGClient:
    """Return (or lazily create) the module-level singleton client."""
    global _client
    if _client is None:
        _client = WeaviateRAGClient()
    return _client
