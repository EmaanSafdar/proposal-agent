# Proposal Assistant

A Slack bot that turns a client discovery-call transcript into a professional DOCX proposal, using past proposals as reference. The consultant can ask follow-up questions, request changes to a single section, and continue the same session on a later day.

## What it does

1. The consultant uploads a transcript (.txt) in a Slack DM with the bot.
2. **Intake agent** extracts structured client data (company, industry, problem, goals, budget, timeline, stakeholders) and marks each field as `stated`, `ambiguous`, `conflicting` or `missing`. Nothing is guessed.
3. **Research agent** searches the past-proposal knowledge base (Chroma vector DB) using tool calling and metadata filters.
4. **Writer agent** plans the structure for this client, then writes each section in a separate LLM call. A **Reviewer agent** fact-checks each section against the intake data and triggers one rewrite if it finds unsupported claims.
5. The bot posts the flagged fields and missing information, then sends the DOCX. Open items are marked `[To confirm: ...]` in red.
6. The consultant can then:
   - ask a question: `what did the client say about their deadline?`
   - request a change: `make the timeline section more detailed` (only that section is rewritten, a new version row is stored)
   - approve: `approve`

## Architecture

```
Slack upload
   |
   v
LangGraph:  Intake  ->  Research  ->  Writer (+ Reviewer per section)
                |            |               |
                v            v               v
           SQLite state   Chroma search   SQLite sections (versioned)
                                                  |
                                                  v
                                            python-docx  ->  Slack file
```

| Component | Choice | Why |
|---|---|---|
| Agent coordination | LangGraph | Explicit data flow, each node testable on its own |
| LLM | Gemini 2.5 Flash (generation) and Gemini embeddings | One provider and one key |
| Vector DB | Chroma (persistent, local) | No server, metadata filtering, simple setup |
| Structured state | SQLite | One file, enough for sessions, versions and messages |
| Slack | Bolt for Python, Socket Mode | Real bot (not a webhook), no public URL needed |
| API | FastAPI (`/health`) | Minimal HTTP surface |
| Documents | python-docx | DOCX generation |

Revision and Q&A are separate paths. A router LLM call classifies each message as `question`, `revise`, `approve` or `other`. A revision rewrites only the chosen section and saves it as a new version; earlier versions are kept as history.

## Requirements

- Python 3.11 or newer
- A Gemini API key (https://aistudio.google.com/apikey)
- A Slack workspace and a Slack app (see below)

## Setup

### 1. Install
```
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install fastapi uvicorn slack-bolt chromadb python-docx langgraph pydantic python-dotenv google-genai requests
```

### 2. Slack app
1. Create an app at https://api.slack.com/apps (Blank app) in your workspace.
2. **Socket Mode**: enable it, create an App-Level Token with scope `connections:write` (starts with `xapp-`).
3. **OAuth & Permissions**, Bot Token Scopes: `chat:write`, `files:read`, `files:write`, `app_mentions:read`, `im:history`, `im:read`, `im:write`.
4. **App Home**: turn on the Messages tab and allow messages from it.
5. **Event Subscriptions**: enable, add bot events `message.im`, `app_mention`, `file_shared`, save.
6. **Install App** to the workspace and copy the Bot User OAuth Token (starts with `xoxb-`). Reinstall if you change scopes or events later.

### 3. Environment variables
Create a `.env` file in the project root:

| Variable | Meaning |
|---|---|
| `GEMINI_API_KEY` | Gemini API key |
| `SLACK_BOT_TOKEN` | Bot token, `xoxb-...` |
| `SLACK_APP_TOKEN` | App-level token, `xapp-...` |

Never commit `.env`.

### 4. Data
Past proposals go in `data/proposals/`, named `proposal_<industry>.txt` (for example `proposal_logistics.txt`). The industry metadata is taken from the file name. The test transcript is `data/transcript/client_transcript.txt`.

## Run

```
python -m src.slack_app
```

On start the app creates `state.db` and ingests the proposals into `chroma_db/` if it is empty (it embeds all sections once, about 80 chunks). When you see `Bolt app is running!`, open the bot in Slack and upload a transcript. Generation takes 1 to 3 minutes. Health check: http://localhost:8000/health

Sessions are keyed by the Slack channel or DM id. Restarting the app keeps all state, and a returning user continues the same session.

## Try it (test script)
1. Upload `client_transcript.txt` in the DM.
2. Read the flagged fields and open the DOCX.
3. Ask: `what did the client say about their deadline?`
4. Request: `make the timeline section more detailed`. A revised DOCX is returned.
5. Check that only that section changed:
```
python -c "import sqlite3; c=sqlite3.connect('state.db'); print(c.execute('select key, max(version), count(*) from sections group by key').fetchall())"
```

## Project layout

```
src/
  llm.py            Gemini helpers (structured JSON output, plain text, retries)
  db.py             SQLite schema and queries (sessions, transcripts, extractions, sections, messages)
  graph.py          LangGraph pipeline: Intake -> Research -> Writer
  revision.py       Message router, Q&A, single-section revision
  docx_builder.py   DOCX generation
  slack_app.py      Slack bot (Bolt, Socket Mode) and FastAPI /health
  agents/
    intake.py       Structured extraction with field status
    research.py     Tool-calling search over past proposals
    writer.py       Planning step and section-by-section writing
    reviewer.py     Fact-check of each section against intake data
  rag/
    store.py        Heading-based chunking, embeddings, Chroma, metadata filtering
data/
  proposals/        10 past proposals (seed data)
  transcript/       Test transcript
```

## RAG details

- Each past proposal is split by heading (80 chunks in total). Each chunk keeps `industry`, `client`, `section_title`, a normalized `section_type` and `source_file` as metadata, and the text is prefixed with industry, client and section for better embeddings.
- Retrieval is vector search (cosine) with optional metadata filters, for example `section_type in [budget, timeline_budget]`. It is not keyword matching.
- The Research agent decides its own queries and filters through a tool.

## Database schema (SQLite)

- `sessions(id, created, status)`
- `transcripts(id, session_id, filename, text, created)`
- `extractions(session_id, intake_json, research_json, plan_json)`
- `sections(id, session_id, key, title, role, position, version, text, created)` (one row per section version)
- `messages(id, session_id, role, content, created)`

## Known limitations

See the decision document for the full list and reasons. In short:
- Intake output is not fully deterministic across runs (field statuses can vary for ambiguous items).
- The Writer can still occasionally present an unverified claim as a fact; the Reviewer reduces this but revised sections are not re-reviewed.
- The transcript is passed to the LLM in full rather than embedded, which is fine for short transcripts only.
- Behaviour of a second transcript upload in an existing session (it is stored, but a full regeneration may be triggered) was not tested.
- Approval and change requests are text messages, not Slack buttons. There are no automated tests.
