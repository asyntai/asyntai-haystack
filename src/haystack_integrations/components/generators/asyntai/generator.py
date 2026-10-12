"""Generators that answer from an Asyntai website agent.

Asyntai reads a website, its product feed and its documents, and keeps that
knowledge current. These generators send a question to that agent and return
its answer, so a Haystack pipeline can answer about a real website without
building or maintaining an index of it.
"""

from typing import Any, Dict, List, Optional

from haystack import component
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_integrations.common.asyntai.base import _AsyntaiComponent
from haystack_integrations.common.asyntai.client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, AsyntaiError


def _reply_meta(payload: Dict[str, Any], website_id: Optional[int]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"session_id": payload.get("session_id")}
    if website_id is not None:
        meta["website_id"] = website_id
    return meta


@component
class AsyntaiGenerator(_AsyntaiComponent):
    """Answers a prompt with the Asyntai agent of one website.

    Usage:

    ```python
    from haystack_integrations.components.generators.asyntai import AsyntaiGenerator

    generator = AsyntaiGenerator()
    result = generator.run(prompt="How long does delivery take?")
    print(result["replies"][0])
    ```

    Pass the same ``session_id`` on later calls to keep the conversation;
    the agent then sees the earlier questions and answers.
    """

    def __init__(
        self,
        api_key: Secret = None,
        website_id: Optional[int] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """
        :param api_key: The Asyntai API key. Defaults to the
            ``ASYNTAI_API_KEY`` environment variable.
        :param website_id: The website agent to ask. Leave empty for the first
            website on the account.
        :param base_url: The Asyntai server.
        :param timeout: Seconds to wait for an answer.
        """
        _AsyntaiComponent.__init__(self, api_key=api_key, website_id=website_id, base_url=base_url, timeout=timeout)

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str, session_id: Optional[str] = None):
        """
        :param prompt: The question.
        :param session_id: Optional conversation id, to keep the chat history.
        :returns: ``replies`` with one answer, and ``meta`` with the session id.
        """
        if not prompt or not prompt.strip():
            raise AsyntaiError("The prompt is empty.")
        payload = self.client.chat(prompt, session_id=session_id, website_id=self.website_id)
        return {
            "replies": [payload.get("response", "")],
            "meta": [_reply_meta(payload, self.website_id)],
        }


@component
class AsyntaiChatGenerator(_AsyntaiComponent):
    """Chat generator backed by the Asyntai agent of one website.

    It sends the last user message to Asyntai and returns the answer as an
    assistant ``ChatMessage``. Asyntai keeps the conversation history on its
    side, keyed by ``session_id``.

    Usage:

    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.asyntai import AsyntaiChatGenerator

    chat = AsyntaiChatGenerator()
    result = chat.run([ChatMessage.from_user("Do you ship to Canada?")])
    print(result["replies"][0].text)
    ```
    """

    def __init__(
        self,
        api_key: Secret = None,
        website_id: Optional[int] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """
        :param api_key: The Asyntai API key. Defaults to the
            ``ASYNTAI_API_KEY`` environment variable.
        :param website_id: The website agent to ask. Leave empty for the first
            website on the account.
        :param base_url: The Asyntai server.
        :param timeout: Seconds to wait for an answer.
        """
        _AsyntaiComponent.__init__(self, api_key=api_key, website_id=website_id, base_url=base_url, timeout=timeout)

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], session_id: Optional[str] = None):
        """
        :param messages: The conversation. The last user message is sent.
        :param session_id: Optional conversation id, to keep the chat history.
        :returns: ``replies`` with one assistant message. Its ``meta`` holds
            the session id to pass on the next call.
        """
        question = None
        for message in reversed(messages or []):
            if message.is_from("user") and message.text and message.text.strip():
                question = message.text
                break
        if question is None:
            raise AsyntaiError("The messages contain no user message with text.")
        payload = self.client.chat(question, session_id=session_id, website_id=self.website_id)
        reply = ChatMessage.from_assistant(
            payload.get("response", ""),
            meta=_reply_meta(payload, self.website_id),
        )
        return {"replies": [reply]}
