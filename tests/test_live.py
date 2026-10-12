"""Live tests against a real Asyntai server over HTTP.

Skipped unless both variables are set:

    ASYNTAI_LIVE_KEY   an API key on the Starter plan or higher
    ASYNTAI_LIVE_URL   the base URL, for example https://asyntai.com

Tests that make Asyntai call an AI model (answers, embeddings) also need
ASYNTAI_LIVE_PAID=1, because each one costs a little money.

    ASYNTAI_LIVE_KEY=... ASYNTAI_LIVE_URL=... pytest tests/test_live.py -v
"""

import os
import uuid

import pytest
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_integrations.common.asyntai import AsyntaiClient, AsyntaiError
from haystack_integrations.components.generators.asyntai import AsyntaiChatGenerator, AsyntaiGenerator
from haystack_integrations.components.writers.asyntai import AsyntaiDocumentWriter, AsyntaiUrlIndexer
from haystack_integrations.tools.asyntai import AsyntaiToolset

KEY = os.environ.get("ASYNTAI_LIVE_KEY", "")
URL = os.environ.get("ASYNTAI_LIVE_URL", "")
PAID = os.environ.get("ASYNTAI_LIVE_PAID") == "1"

pytestmark = pytest.mark.skipif(
    not (KEY and URL), reason="set ASYNTAI_LIVE_KEY and ASYNTAI_LIVE_URL to run the live tests"
)
paid = pytest.mark.skipif(not PAID, reason="set ASYNTAI_LIVE_PAID=1 to run tests that call an AI model")

SECRET = Secret.from_env_var("ASYNTAI_LIVE_KEY")


def tool(toolset, name):
    return next(t for t in toolset if t.name == name)


# ----------------------------------------------------------------------
# free: no AI model is called
# ----------------------------------------------------------------------
def test_websites_list():
    sites = AsyntaiClient(api_key=KEY, base_url=URL).websites()
    assert sites and "id" in sites[0] and "domain" in sites[0]


def test_bad_key_is_rejected():
    generator = AsyntaiGenerator(api_key=Secret.from_token("definitely-not-a-key"), base_url=URL)
    with pytest.raises(AsyntaiError, match="key"):
        generator.run(prompt="Hello there")


def test_unknown_website_is_explained():
    generator = AsyntaiGenerator(api_key=SECRET, base_url=URL, website_id=999999999)
    with pytest.raises(AsyntaiError) as exc:
        generator.run(prompt="Hello there")
    assert "website" in str(exc.value).lower()


def test_read_tools_answer():
    ts = AsyntaiToolset(api_key=SECRET, base_url=URL)
    assert isinstance(tool(ts, "asyntai_list_knowledge").invoke(limit=5), str)
    assert isinstance(tool(ts, "asyntai_get_leads").invoke(limit=5), str)
    assert "No messages" in tool(ts, "asyntai_get_conversation").invoke(session_id=f"none-{uuid.uuid4().hex}")


def test_short_documents_are_skipped_without_a_call():
    out = AsyntaiDocumentWriter(api_key=SECRET, base_url=URL).run(documents=[Document(content="tiny")])
    assert out == {"documents_written": 0, "skipped": [out["skipped"][0]], "failed": []}


def test_unreachable_url_is_reported():
    out = AsyntaiUrlIndexer(api_key=SECRET, base_url=URL, raise_on_failure=False).run(
        urls=["https://does-not-exist.invalid/page"])
    assert out["indexed"] == []
    assert out["failed"] and out["failed"][0]["url"] == "https://does-not-exist.invalid/page"


# ----------------------------------------------------------------------
# paid: Asyntai embeds texts and answers with an AI model
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def marker():
    return uuid.uuid4().hex[:8]


@paid
def test_indexing_pipeline_teaches_the_agent(tmp_path, marker):
    path = tmp_path / f"hours-{marker}.txt"
    path.write_text(
        f"The Haystack live test shop is open Monday to Friday, 9 to 17. "
        f"Its returns code is RET-{marker}.", encoding="utf-8")
    pipeline = Pipeline()
    pipeline.add_component("converter", TextFileToDocument())
    pipeline.add_component("writer", AsyntaiDocumentWriter(api_key=SECRET, base_url=URL))
    pipeline.connect("converter.documents", "writer.documents")
    out = pipeline.run({"converter": {"sources": [path]}})
    assert out["writer"]["documents_written"] == 1

    listing = tool(AsyntaiToolset(api_key=SECRET, base_url=URL), "asyntai_list_knowledge").invoke(limit=20)
    assert f"hours-{marker}.txt" in listing


@paid
def test_generator_answers_from_the_new_knowledge(marker):
    pipeline = Pipeline()
    pipeline.add_component("prompt", PromptBuilder(template="{{ question }}", required_variables=["question"]))
    pipeline.add_component("llm", AsyntaiGenerator(api_key=SECRET, base_url=URL))
    pipeline.connect("prompt.prompt", "llm.prompt")
    out = pipeline.run({"prompt": {"question": "What is the returns code of the Haystack live test shop?"}})
    reply = out["llm"]["replies"][0]
    assert f"RET-{marker}" in reply, reply
    assert out["llm"]["meta"][0]["session_id"]


@paid
def test_chat_generator_keeps_the_session(marker):
    chat = AsyntaiChatGenerator(api_key=SECRET, base_url=URL)
    session = f"haystack-live-{marker}"
    first = chat.run([ChatMessage.from_user("On which days is the Haystack live test shop open?")],
                     session_id=session)
    assert "Monday" in first["replies"][0].text, first["replies"][0].text
    history = tool(AsyntaiToolset(api_key=SECRET, base_url=URL), "asyntai_get_conversation").invoke(
        session_id=session)
    assert "Visitor: On which days" in history
