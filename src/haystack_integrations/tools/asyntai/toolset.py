"""Asyntai tools for Haystack agents.

``AsyntaiToolset`` gives a Haystack ``Agent`` six tools on the Asyntai API:
ask the website agent, teach it a text or a page, list what it knows, read
the leads it collected and read one chat.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool, Toolset
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.common.asyntai.base import _AsyntaiComponent
from haystack_integrations.common.asyntai.client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT

TOOL_NAMES = (
    "asyntai_ask_website",
    "asyntai_add_text",
    "asyntai_add_url",
    "asyntai_list_knowledge",
    "asyntai_get_leads",
    "asyntai_get_conversation",
)


def _clamp(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(100, number))


class AsyntaiToolset(Toolset):
    """The Asyntai API as tools for a Haystack ``Agent``.

    Usage:

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.tools.asyntai import AsyntaiToolset

    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        tools=AsyntaiToolset(),
    )
    result = agent.run([ChatMessage.from_user("Who asked about delivery this week?")])
    print(result["messages"][-1].text)
    ```

    Pass ``tools=[...]`` with names from ``TOOL_NAMES`` to give the agent
    only some of the tools, for example only ``asyntai_ask_website``.
    """

    def __init__(
        self,
        api_key: Secret = None,
        website_id: Optional[int] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        tools: Optional[List[str]] = None,
    ) -> None:
        """
        :param api_key: The Asyntai API key. Defaults to the
            ``ASYNTAI_API_KEY`` environment variable.
        :param website_id: The website agent to use. Leave empty for the
            first website on the account.
        :param base_url: The Asyntai server.
        :param timeout: Seconds to wait for each call.
        :param tools: Names of the tools to include. All six when empty.
        """
        self._settings = _AsyntaiComponent(
            api_key=api_key, website_id=website_id, base_url=base_url, timeout=timeout
        )
        unknown = [name for name in (tools or []) if name not in TOOL_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown Asyntai tool name(s): {', '.join(unknown)}. "
                f"Choose from: {', '.join(TOOL_NAMES)}."
            )
        self.tool_names = list(tools) if tools else list(TOOL_NAMES)
        built = self._build_tools()
        super().__init__(tools=[built[name] for name in self.tool_names])

    # ------------------------------------------------------------------
    @property
    def website_id(self) -> Optional[int]:
        return self._settings.website_id

    @property
    def client(self):
        return self._settings.client

    def warm_up(self) -> None:
        self._settings.warm_up()
        super().warm_up()

    # ------------------------------------------------------------------
    def _ask(self, question: str, session_id: Optional[str] = None) -> str:
        payload = self.client.chat(question, session_id=session_id, website_id=self.website_id)
        return payload.get("response", "")

    def _add_text(self, title: str, content: str) -> str:
        payload = self.client.add_text(title=title, content=content, website_id=self.website_id)
        chunks = payload.get("chunks_created", 0)
        return f"Stored '{payload.get('title', title)}' as {chunks} chunks."

    def _add_url(self, url: str) -> str:
        payload = self.client.add_url(url=url, website_id=self.website_id)
        chunks = payload.get("chunks_created", 0)
        return f"Read '{payload.get('title', url)}' into the knowledge base as {chunks} chunks."

    def _list_knowledge(self, limit: int = 20) -> str:
        entries = self.client.knowledge(limit=_clamp(limit, 20), website_id=self.website_id)
        if not entries:
            return "The knowledge base has no entries yet."
        return "\n".join(
            f"- {e.get('title', 'Untitled')} ({e.get('type', 'text')}, "
            f"{e.get('chunks_count', 0)} chunks)"
            for e in entries
        )

    def _leads(self, limit: int = 20, email: Optional[str] = None, since: Optional[str] = None) -> str:
        leads = self.client.leads(
            limit=_clamp(limit, 20), website_id=self.website_id, email=email or None, since=since or None
        )
        if not leads:
            return "No leads match that request."
        return json.dumps(leads, ensure_ascii=False, indent=2)

    def _conversation(self, session_id: str, limit: int = 50) -> str:
        messages = self.client.conversation(session_id=session_id, limit=_clamp(limit, 50))
        if not messages:
            return f"No messages found for session {session_id}."
        return "\n".join(
            f"{'Visitor' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
            for m in messages
        )

    def _build_tools(self) -> Dict[str, Tool]:
        def tool(name: str, description: str, properties: Dict[str, Any],
                 required: List[str], function: Callable[..., str]) -> Tool:
            return Tool(
                name=name,
                description=description,
                parameters={"type": "object", "properties": properties, "required": required},
                function=function,
            )

        limit = lambda default: {  # noqa: E731
            "type": "integer",
            "description": f"How many to return, 1 to 100. Default {default}.",
        }
        return {
            "asyntai_ask_website": tool(
                "asyntai_ask_website",
                "Answer a question about a company website: products, prices, delivery, "
                "opening hours, policies, documentation and any page Asyntai has read. "
                "Use this whenever the answer should come from the website rather than "
                "from general knowledge.",
                {
                    "question": {"type": "string", "description": "The question in plain language."},
                    "session_id": {
                        "type": "string",
                        "description": "Optional conversation id, to keep earlier questions in view.",
                    },
                },
                ["question"],
                self._ask,
            ),
            "asyntai_add_text": tool(
                "asyntai_add_text",
                "Store a text in the Asyntai knowledge base so the website agent can use "
                "it in later answers.",
                {
                    "title": {"type": "string", "description": "A short title for this piece of knowledge."},
                    "content": {"type": "string", "description": "The text to store. At least 10 characters."},
                },
                ["title", "content"],
                self._add_text,
            ),
            "asyntai_add_url": tool(
                "asyntai_add_url",
                "Read a web page and store it in the Asyntai knowledge base, so the "
                "website agent can answer from it.",
                {"url": {"type": "string", "description": "The address of the page to read."}},
                ["url"],
                self._add_url,
            ),
            "asyntai_list_knowledge": tool(
                "asyntai_list_knowledge",
                "List the entries in the Asyntai knowledge base, newest first. Use it to "
                "check what the website agent already knows before you add more.",
                {"limit": limit(20)},
                [],
                self._list_knowledge,
            ),
            "asyntai_get_leads": tool(
                "asyntai_get_leads",
                "List the website visitors who left an email address or a phone number "
                "in the chat. Returns the address, the page they were on and the time.",
                {
                    "limit": limit(20),
                    "email": {"type": "string", "description": "Only the leads with this email address."},
                    "since": {
                        "type": "string",
                        "description": "ISO date, for example 2026-01-15. Only leads from that day on.",
                    },
                },
                [],
                self._leads,
            ),
            "asyntai_get_conversation": tool(
                "asyntai_get_conversation",
                "Read the messages of one website chat, oldest first. The lead list "
                "returns the session id.",
                {
                    "session_id": {"type": "string", "description": "The id of the chat session to read."},
                    "limit": limit(50),
                },
                ["session_id"],
                self._conversation,
            ),
        }

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = self._settings._init_params()
        data["tools"] = list(self.tool_names)
        return {"type": generate_qualified_class_name(type(self)), "data": data}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AsyntaiToolset":
        params = dict(data["data"])
        deserialize_secrets_inplace(params, keys=["api_key"])
        return cls(**params)
