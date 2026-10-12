"""Shared settings for every Asyntai component.

Each component keeps its API key as a Haystack ``Secret`` so a pipeline can be
saved to YAML without writing the key into the file, and builds its HTTP
client on first use so a pipeline loads even before the key is set.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from haystack import default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from .client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, AsyntaiClient


def default_api_key() -> Secret:
    return Secret.from_env_var("ASYNTAI_API_KEY")


class _AsyntaiComponent:
    """Mixin: API key, base URL, website and a lazily built client."""

    def __init__(
        self,
        api_key: Secret = None,
        website_id: Optional[int] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key if api_key is not None else default_api_key()
        self.website_id = website_id
        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[AsyntaiClient] = None

    @property
    def client(self) -> AsyntaiClient:
        if self._client is None:
            self._client = AsyntaiClient(
                api_key=self.api_key.resolve_value(),
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def warm_up(self) -> None:
        """Build the client up front, so a missing key fails at start-up."""
        _ = self.client

    def _init_params(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key.to_dict(),
            "website_id": self.website_id,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, **self._init_params())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)
