"""Unit tests. No network: every HTTP call goes to requests_mock."""

import json

import pytest
import requests_mock as rm
from haystack import Document, Pipeline, component
from haystack.components.agents import Agent
from haystack.components.builders import PromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.utils import Secret

from haystack_integrations.common.asyntai import AsyntaiError
from haystack_integrations.components.generators.asyntai import AsyntaiChatGenerator, AsyntaiGenerator
from haystack_integrations.components.writers.asyntai import AsyntaiDocumentWriter, AsyntaiUrlIndexer
from haystack_integrations.components.writers.asyntai.writer import document_title
from haystack_integrations.tools.asyntai import TOOL_NAMES, AsyntaiToolset

BASE = "https://asyntai.test"
KEY = "unit-test-key"


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("ASYNTAI_API_KEY", KEY)


@pytest.fixture
def api():
    with rm.Mocker() as m:
        yield m


def ok(**extra):
    body = {"success": True}
    body.update(extra)
    return body


def sent(m, n=-1):
    return json.loads(m.request_history[n].body)


# ----------------------------------------------------------------------
# generators
# ----------------------------------------------------------------------
def test_generator_returns_the_answer_and_session(api):
    api.post(f"{BASE}/api/v1/chat/", json=ok(response="Delivery takes 2 days.", session_id="s1"))
    out = AsyntaiGenerator(base_url=BASE).run(prompt="How long is delivery?")
    assert out["replies"] == ["Delivery takes 2 days."]
    assert out["meta"] == [{"session_id": "s1"}]
    req = api.request_history[0]
    assert req.headers["Authorization"] == f"Bearer {KEY}"
    assert req.headers["User-Agent"] == "asyntai-haystack"
    assert sent(api) == {"message": "How long is delivery?"}


def test_generator_sends_session_and_website(api):
    api.post(f"{BASE}/api/v1/chat/", json=ok(response="Yes.", session_id="abc"))
    out = AsyntaiGenerator(base_url=BASE, website_id=42).run(prompt="Open on Sunday?", session_id="abc")
    assert sent(api) == {"message": "Open on Sunday?", "session_id": "abc", "website_id": 42}
    assert out["meta"] == [{"session_id": "abc", "website_id": 42}]


def test_generator_refuses_an_empty_prompt(api):
    with pytest.raises(AsyntaiError):
        AsyntaiGenerator(base_url=BASE).run(prompt="   ")
    assert not api.called


@pytest.mark.parametrize("status,word", [(401, "key"), (403, "Starter"), (429, "limit")])
def test_generator_explains_refusals(api, status, word):
    api.post(f"{BASE}/api/v1/chat/", status_code=status, json={"success": False, "error": "x"})
    with pytest.raises(AsyntaiError) as exc:
        AsyntaiGenerator(base_url=BASE).run(prompt="Hi there")
    assert word in str(exc.value)


def test_api_error_text_is_passed_on(api):
    api.post(f"{BASE}/api/v1/chat/", status_code=400, json={"success": False, "error": "Website not found"})
    with pytest.raises(AsyntaiError, match="Website not found"):
        AsyntaiGenerator(base_url=BASE, website_id=9).run(prompt="Hi there")


def test_non_json_body_is_an_asyntai_error(api):
    api.post(f"{BASE}/api/v1/chat/", status_code=502, text="<html>Bad gateway</html>")
    with pytest.raises(AsyntaiError, match="non-JSON"):
        AsyntaiGenerator(base_url=BASE).run(prompt="Hi there")


def test_missing_key_fails_on_first_use_not_on_build(monkeypatch):
    monkeypatch.delenv("ASYNTAI_API_KEY")
    generator = AsyntaiGenerator(base_url=BASE)  # building must not need the key
    with pytest.raises(ValueError):
        generator.warm_up()


def test_explicit_secret_wins_over_the_environment(api):
    api.post(f"{BASE}/api/v1/chat/", json=ok(response="Hi", session_id="s"))
    AsyntaiGenerator(api_key=Secret.from_token("tok-123"), base_url=BASE).run(prompt="Hello")
    assert api.request_history[0].headers["Authorization"] == "Bearer tok-123"


def test_chat_generator_sends_the_last_user_message(api):
    api.post(f"{BASE}/api/v1/chat/", json=ok(response="We ship to Canada.", session_id="s9"))
    messages = [
        ChatMessage.from_system("Be brief."),
        ChatMessage.from_user("Hello"),
        ChatMessage.from_assistant("Hi, how can I help?"),
        ChatMessage.from_user("Do you ship to Canada?"),
    ]
    out = AsyntaiChatGenerator(base_url=BASE).run(messages=messages)
    assert sent(api) == {"message": "Do you ship to Canada?"}
    reply = out["replies"][0]
    assert reply.is_from("assistant")
    assert reply.text == "We ship to Canada."
    assert reply.meta == {"session_id": "s9"}


def test_chat_generator_needs_a_user_message(api):
    with pytest.raises(AsyntaiError):
        AsyntaiChatGenerator(base_url=BASE).run(messages=[ChatMessage.from_system("x")])
    assert not api.called


# ----------------------------------------------------------------------
# writers
# ----------------------------------------------------------------------
def test_document_title_order():
    assert document_title(Document(content="x", meta={"title": " Returns "})) == "Returns"
    assert document_title(Document(content="x", meta={"file_path": "C:\\docs\\faq.txt"})) == "faq.txt"
    assert document_title(Document(content="x", meta={"file_path": "/srv/docs/faq.md"})) == "faq.md"
    assert document_title(Document(content="x", meta={"url": "https://a.test/shipping/"})) == "shipping"
    doc = Document(content="plain text")
    assert document_title(doc) == f"Document {doc.id[:12]}"
    assert len(document_title(Document(content="x", meta={"title": "t" * 500}))) == 200


def test_writer_stores_each_document(api):
    api.post(f"{BASE}/api/v1/knowledge/text/", json=ok(chunks_created=1))
    docs = [
        Document(content="Returns are free within 30 days.", meta={"title": "Returns"}),
        Document(content="We deliver in 2 to 3 working days.", meta={"file_path": "delivery.txt"}),
        Document(content="short"),
        Document(content=None),
    ]
    out = AsyntaiDocumentWriter(base_url=BASE, website_id=7).run(documents=docs)
    assert out["documents_written"] == 2
    assert out["skipped"] == [docs[2].id, docs[3].id]
    assert out["failed"] == []
    assert sent(api, 0) == {"title": "Returns", "content": "Returns are free within 30 days.", "website_id": 7}
    assert sent(api, 1)["title"] == "delivery.txt"


def test_writer_raises_by_default(api):
    api.post(f"{BASE}/api/v1/knowledge/text/", status_code=400,
             json={"success": False, "error": "Knowledge base limit reached"})
    with pytest.raises(AsyntaiError, match="limit reached"):
        AsyntaiDocumentWriter(base_url=BASE).run(documents=[Document(content="A long enough text.")])


def test_writer_can_continue_past_failures(api):
    api.post(f"{BASE}/api/v1/knowledge/text/", [
        {"status_code": 400, "json": {"success": False, "error": "Too big"}},
        {"json": ok(chunks_created=2)},
    ])
    docs = [Document(content="First long enough text."), Document(content="Second long enough text.")]
    out = AsyntaiDocumentWriter(base_url=BASE, raise_on_failure=False).run(documents=docs)
    assert out["documents_written"] == 1
    assert out["failed"] == [{"id": docs[0].id, "title": f"Document {docs[0].id[:12]}", "error": "Too big"}]


def test_url_indexer(api):
    api.post(f"{BASE}/api/v1/knowledge/url/", [
        {"json": ok(title="Shipping", chunks_created=4)},
        {"status_code": 400, "json": {"success": False, "error": "Could not fetch the page"}},
    ])
    out = AsyntaiUrlIndexer(base_url=BASE, raise_on_failure=False).run(
        urls=["https://a.test/shipping", "https://a.test/missing"])
    assert out["indexed"] == [{"url": "https://a.test/shipping", "title": "Shipping", "chunks": 4}]
    assert out["failed"] == [{"url": "https://a.test/missing", "error": "Could not fetch the page"}]
    assert sent(api, 0) == {"url": "https://a.test/shipping"}


# ----------------------------------------------------------------------
# toolset
# ----------------------------------------------------------------------
def test_toolset_has_six_tools_with_valid_schemas():
    toolset = AsyntaiToolset(base_url=BASE)
    assert [t.name for t in toolset] == list(TOOL_NAMES)
    for tool in toolset:
        assert tool.parameters["type"] == "object"
        assert set(tool.parameters["required"]) <= set(tool.parameters["properties"])
        assert tool.description


def test_toolset_subset_and_unknown_name():
    assert [t.name for t in AsyntaiToolset(tools=["asyntai_ask_website"])] == ["asyntai_ask_website"]
    with pytest.raises(ValueError, match="asyntai_nope"):
        AsyntaiToolset(tools=["asyntai_nope"])


def tool(toolset, name):
    return next(t for t in toolset if t.name == name)


def test_each_tool_calls_its_endpoint(api):
    api.post(f"{BASE}/api/v1/chat/", json=ok(response="Open 9 to 17.", session_id="s"))
    api.post(f"{BASE}/api/v1/knowledge/text/", json=ok(title="Hours", chunks_created=1))
    api.post(f"{BASE}/api/v1/knowledge/url/", json=ok(title="FAQ", chunks_created=3))
    api.get(f"{BASE}/api/v1/knowledge/", json=ok(entries=[{"title": "FAQ", "type": "url", "chunks_count": 3}]))
    api.get(f"{BASE}/api/v1/leads/", json=ok(leads=[{"email": "a@b.test", "page_url": "/"}]))
    api.get(f"{BASE}/api/v1/conversations/", json=ok(messages=[
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]))

    ts = AsyntaiToolset(base_url=BASE, website_id=5)
    assert tool(ts, "asyntai_ask_website").invoke(question="Hours?") == "Open 9 to 17."
    assert sent(api)["website_id"] == 5
    assert tool(ts, "asyntai_add_text").invoke(title="Hours", content="Open 9 to 17 daily.") == \
        "Stored 'Hours' as 1 chunks."
    assert "FAQ" in tool(ts, "asyntai_add_url").invoke(url="https://a.test/faq")
    assert tool(ts, "asyntai_list_knowledge").invoke(limit=500) == "- FAQ (url, 3 chunks)"
    assert api.last_request.qs["limit"] == ["100"]
    assert "a@b.test" in tool(ts, "asyntai_get_leads").invoke(since="2026-01-01")
    assert api.last_request.qs["since"] == ["2026-01-01"]
    assert tool(ts, "asyntai_get_conversation").invoke(session_id="s") == "Visitor: Hi\nAssistant: Hello"


def test_empty_results_read_well(api):
    api.get(f"{BASE}/api/v1/knowledge/", json=ok(entries=[]))
    api.get(f"{BASE}/api/v1/leads/", json=ok(leads=[]))
    api.get(f"{BASE}/api/v1/conversations/", json=ok(messages=[]))
    ts = AsyntaiToolset(base_url=BASE)
    assert "no entries" in tool(ts, "asyntai_list_knowledge").invoke()
    assert "No leads" in tool(ts, "asyntai_get_leads").invoke()
    assert "No messages" in tool(ts, "asyntai_get_conversation").invoke(session_id="x")


# ----------------------------------------------------------------------
# serialization: the key must never be written out
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", [AsyntaiGenerator, AsyntaiChatGenerator, AsyntaiDocumentWriter, AsyntaiUrlIndexer])
def test_component_round_trip(cls):
    original = cls(base_url=BASE, website_id=3, timeout=12)
    data = original.to_dict()
    assert KEY not in json.dumps(data)
    assert data["init_parameters"]["api_key"] == {"type": "env_var", "env_vars": ["ASYNTAI_API_KEY"], "strict": True}
    copy = cls.from_dict(data)
    assert (copy.base_url, copy.website_id, copy.timeout) == (BASE, 3, 12)


def test_token_secret_cannot_be_serialized():
    with pytest.raises(ValueError):
        AsyntaiGenerator(api_key=Secret.from_token("tok")).to_dict()


def test_toolset_round_trip():
    ts = AsyntaiToolset(base_url=BASE, website_id=4, tools=["asyntai_ask_website", "asyntai_get_leads"])
    data = ts.to_dict()
    assert KEY not in json.dumps(data)
    copy = AsyntaiToolset.from_dict(data)
    assert [t.name for t in copy] == ["asyntai_ask_website", "asyntai_get_leads"]
    assert copy.website_id == 4


# ----------------------------------------------------------------------
# real Haystack pipelines
# ----------------------------------------------------------------------
def test_indexing_pipeline_from_files(api, tmp_path):
    api.post(f"{BASE}/api/v1/knowledge/text/", json=ok(chunks_created=1))
    (tmp_path / "returns.txt").write_text("Returns are free within 30 days of delivery.", encoding="utf-8")
    (tmp_path / "hours.txt").write_text("The shop is open Monday to Friday, 9 to 17.", encoding="utf-8")

    pipeline = Pipeline()
    pipeline.add_component("converter", TextFileToDocument())
    pipeline.add_component("writer", AsyntaiDocumentWriter(base_url=BASE))
    pipeline.connect("converter.documents", "writer.documents")
    out = pipeline.run({"converter": {"sources": [tmp_path / "returns.txt", tmp_path / "hours.txt"]}})

    assert out["writer"]["documents_written"] == 2
    assert {sent(api, 0)["title"], sent(api, 1)["title"]} == {"returns.txt", "hours.txt"}


def test_question_pipeline_and_yaml_round_trip(api):
    api.post(f"{BASE}/api/v1/chat/", json=ok(response="Returns are free.", session_id="s"))
    pipeline = Pipeline()
    pipeline.add_component("prompt", PromptBuilder(template="Customer asks: {{ question }}",
                                                   required_variables=["question"]))
    pipeline.add_component("llm", AsyntaiGenerator(base_url=BASE))
    pipeline.connect("prompt.prompt", "llm.prompt")

    yaml_text = pipeline.dumps()
    assert KEY not in yaml_text
    loaded = Pipeline.loads(yaml_text)
    out = loaded.run({"prompt": {"question": "Are returns free?"}})
    assert out["llm"]["replies"] == ["Returns are free."]
    assert sent(api)["message"] == "Customer asks: Are returns free?"


@component
class ScriptedChatGenerator:
    """A chat generator that plays back fixed replies: a tool call, then text."""

    def __init__(self):
        self.calls = []

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools=None, **kwargs):
        self.calls.append((list(messages), [t.name for t in (tools or [])]))
        if len(self.calls) == 1:
            call = ToolCall(tool_name="asyntai_ask_website", arguments={"question": "Do you ship to Canada?"})
            return {"replies": [ChatMessage.from_assistant(tool_calls=[call])]}
        result = messages[-1].tool_call_result.result
        return {"replies": [ChatMessage.from_assistant(f"The website says: {result}")]}


def test_agent_calls_an_asyntai_tool(api):
    api.post(f"{BASE}/api/v1/chat/", json=ok(response="Yes, we ship to Canada.", session_id="s"))
    llm = ScriptedChatGenerator()
    agent = Agent(chat_generator=llm, tools=AsyntaiToolset(base_url=BASE))
    agent.warm_up()
    out = agent.run(messages=[ChatMessage.from_user("Can I order to Toronto?")])

    assert out["messages"][-1].text == "The website says: Yes, we ship to Canada."
    assert sent(api) == {"message": "Do you ship to Canada?"}
    assert set(llm.calls[0][1]) == set(TOOL_NAMES)


def test_agent_pipeline_with_the_toolset_saves_and_loads(monkeypatch):
    from haystack.components.generators.chat import OpenAIChatGenerator

    monkeypatch.setenv("OPENAI_API_KEY", "not-used")  # the pipeline is only saved and loaded
    pipeline = Pipeline()
    pipeline.add_component("agent", Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=AsyntaiToolset(base_url=BASE, website_id=8, tools=["asyntai_ask_website"]),
    ))
    yaml_text = pipeline.dumps()
    assert KEY not in yaml_text
    assert "haystack_integrations.tools.asyntai.toolset.AsyntaiToolset" in yaml_text

    loaded = Pipeline.loads(yaml_text)  # default allowlist, no unsafe flag
    toolset = loaded.get_component("agent").tools
    assert isinstance(toolset, AsyntaiToolset)
    assert [t.name for t in toolset] == ["asyntai_ask_website"]
    assert toolset.website_id == 8
