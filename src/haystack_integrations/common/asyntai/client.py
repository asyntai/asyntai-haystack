"""Thin HTTP client for the Asyntai public API.

Every component in this package goes through one client so that authentication,
timeouts and error messages have a single implementation. The API is
documented at https://asyntai.com/documentation/api-reference/.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE_URL = "https://asyntai.com"
DEFAULT_TIMEOUT = 60


class AsyntaiError(RuntimeError):
    """Raised when the Asyntai API refuses a call or cannot be reached."""


class AsyntaiClient:
    """Calls the Asyntai REST API with a single API key.

    The key is read from ``api_key`` or, when that is empty, from the
    ``ASYNTAI_API_KEY`` environment variable. Keys are created in the Asyntai
    dashboard under Settings -> API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        key = api_key or os.environ.get("ASYNTAI_API_KEY", "")
        if not key:
            raise AsyntaiError(
                "An Asyntai API key is required. Pass api_key=... or set the "
                "ASYNTAI_API_KEY environment variable. Create a key at "
                "https://asyntai.com/ under Settings -> API."
            )
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "asyntai-haystack",
        }
        try:
            response = self.session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise AsyntaiError(f"Could not reach the Asyntai API: {exc}") from exc

        try:
            payload = response.json()
        except ValueError:
            raise AsyntaiError(
                f"Asyntai returned a non-JSON response (HTTP {response.status_code})."
            ) from None

        if not isinstance(payload, dict):
            raise AsyntaiError("Asyntai returned an unexpected response body.")

        if response.status_code == 401:
            raise AsyntaiError(
                "Asyntai rejected the API key. Check the key in Settings -> API."
            )
        if response.status_code == 403:
            raise AsyntaiError(
                "The Asyntai API needs the Starter plan or higher. "
                "See https://asyntai.com/pricing/."
            )
        if response.status_code == 429:
            raise AsyntaiError(
                "The Asyntai message limit for this month is reached."
            )
        if not payload.get("success", False):
            raise AsyntaiError(
                payload.get("error") or f"Asyntai API error (HTTP {response.status_code})."
            )
        return payload

    # ------------------------------------------------------------------
    # endpoints
    # ------------------------------------------------------------------
    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        website_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """POST /api/v1/chat/ — ask the agent a question."""
        body: Dict[str, Any] = {"message": message}
        if session_id:
            body["session_id"] = session_id
        if website_id is not None:
            body["website_id"] = website_id
        return self._request("POST", "/api/v1/chat/", json_body=body)

    def websites(self) -> List[Dict[str, Any]]:
        """GET /api/v1/websites/ — the agents on this account."""
        return self._request("GET", "/api/v1/websites/").get("websites", [])

    def add_text(
        self,
        title: str,
        content: str,
        website_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """POST /api/v1/knowledge/text/ — store a text in the knowledge base."""
        body: Dict[str, Any] = {"title": title, "content": content}
        if website_id is not None:
            body["website_id"] = website_id
        return self._request("POST", "/api/v1/knowledge/text/", json_body=body)

    def add_url(
        self,
        url: str,
        website_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """POST /api/v1/knowledge/url/ — read a page into the knowledge base."""
        body: Dict[str, Any] = {"url": url}
        if website_id is not None:
            body["website_id"] = website_id
        return self._request("POST", "/api/v1/knowledge/url/", json_body=body)

    def knowledge(
        self,
        limit: int = 50,
        website_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """GET /api/v1/knowledge/ — what the agent knows."""
        params: Dict[str, Any] = {"limit": limit}
        if website_id is not None:
            params["website_id"] = website_id
        return self._request("GET", "/api/v1/knowledge/", params=params).get("entries", [])

    def leads(
        self,
        limit: int = 50,
        website_id: Optional[int] = None,
        email: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """GET /api/v1/leads/ — the visitors who left an address."""
        params: Dict[str, Any] = {"limit": limit}
        if website_id is not None:
            params["website_id"] = website_id
        if email:
            params["email"] = email
        if since:
            params["since"] = since
        return self._request("GET", "/api/v1/leads/", params=params).get("leads", [])

    def conversation(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """GET /api/v1/conversations/ — the messages of one chat."""
        params = {"session_id": session_id, "limit": limit}
        payload = self._request("GET", "/api/v1/conversations/", params=params)
        return payload.get("messages", [])
