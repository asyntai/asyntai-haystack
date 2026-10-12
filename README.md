# asyntai-haystack

[Haystack](https://haystack.deepset.ai) components and agent tools for
[Asyntai](https://asyntai.com), the AI chat agent for websites.

Point Asyntai at a website. It reads the pages, the product feed and the
documents, and keeps them current. This package brings that agent into
Haystack: ask it questions inside a pipeline, feed it documents from an
indexing pipeline, and give a Haystack `Agent` tools to read its leads and
chats.

## Install

```bash
pip install asyntai-haystack
```

## Set the key

Create an API key in Asyntai under **Settings → API**. The API is available on
the Starter plan and higher.

```bash
export ASYNTAI_API_KEY="your-key"
```

## Components

| Component | What it does |
| --- | --- |
| `AsyntaiGenerator` | Answers a prompt from the website: products, prices, delivery, policies, documentation. |
| `AsyntaiChatGenerator` | The same, as a chat generator that takes and returns `ChatMessage` objects. |
| `AsyntaiDocumentWriter` | Stores Haystack documents in the website agent's knowledge base. |
| `AsyntaiUrlIndexer` | Asks Asyntai to read web pages into the knowledge base. |
| `AsyntaiToolset` | Six tools for a Haystack `Agent`. |

## Ask the website a question

```python
from haystack_integrations.components.generators.asyntai import AsyntaiGenerator

generator = AsyntaiGenerator()
result = generator.run(prompt="How long does delivery take?")
print(result["replies"][0])
```

Pass the `session_id` from `result["meta"][0]` on the next call to keep the
conversation going.

## Teach the website agent from your files

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

The title in Asyntai comes from `meta["title"]`, then the file name. Any
converter works in front of the writer: PDF, DOCX, HTML, Markdown.

To read live pages instead:

```python
from haystack_integrations.components.writers.asyntai import AsyntaiUrlIndexer

AsyntaiUrlIndexer().run(urls=["https://example.com/shipping"])
```

## Give an agent the Asyntai tools

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

| Tool | What the agent can do |
| --- | --- |
| `asyntai_ask_website` | Answer a question from the website. |
| `asyntai_add_text` | Store a text so the website agent uses it in later answers. |
| `asyntai_add_url` | Read a page into the knowledge base. |
| `asyntai_list_knowledge` | List what the website agent knows. |
| `asyntai_get_leads` | Read the visitors who left an email address or a phone number. |
| `asyntai_get_conversation` | Read the messages of one website chat. |

Give the agent only some of them with
`AsyntaiToolset(tools=["asyntai_ask_website"])`.

## Several websites on one account

An Asyntai account can run an agent per website. Name the one you want with
`website_id`, which `AsyntaiClient().websites()` returns.

```python
from haystack_integrations.common.asyntai import AsyntaiClient
from haystack_integrations.components.generators.asyntai import AsyntaiGenerator

for site in AsyntaiClient().websites():
    print(site["id"], site["domain"])

generator = AsyntaiGenerator(website_id=42)
```

## Save and load pipelines

Every component serializes to YAML with `pipeline.dumps()`. The API key is
stored as a reference to the `ASYNTAI_API_KEY` environment variable, never as
the key itself.

## Errors

Every failure raises `AsyntaiError` with a message you can show to a user.
The writer and the URL indexer take `raise_on_failure=False` to log a refused
item and continue.

## Links

- Asyntai: https://asyntai.com
- Documentation: https://asyntai.com/documentation/integrations/haystack/
- API reference: https://asyntai.com/documentation/api-reference/

## License

MIT
