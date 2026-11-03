import os
import tempfile
import threading
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from src import db
from src.docx_builder import build_docx
from src.graph import generate_proposal
from src.rag.store import ingest_if_empty
from src.revision import answer_question, revise_section, route

load_dotenv()
app = App(token=os.environ["SLACK_BOT_TOKEN"])
api = FastAPI()  # minimal HTTP surface (health check)


@api.get("/health")
def health():
    return {"ok": True}


def send_docx(client, channel, sid, comment):
    secs = db.current_sections(sid)
    path = Path(tempfile.gettempdir()) / f"proposal_{sid}.docx"
    build_docx(secs, {s["key"]: s["text"] for s in secs}, path)
    client.files_upload_v2(channel=channel, file=str(path), filename="proposal.docx", initial_comment=comment)


def process_upload(client, channel, sid, files):
    for f in files:
        info = client.files_info(file=f["id"])["file"]
        r = requests.get(info["url_private_download"],
                         headers={"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"}, timeout=60)
        r.raise_for_status()
        db.add_transcript(sid, info["name"], r.content.decode("utf-8", errors="replace"))
    client.chat_postMessage(channel=channel, text="Transcript received. Analysing it and drafting the proposal, "
                                                  "this takes a minute or two...")
    try:
        generate_proposal(sid)
        ex = db.get_extraction(sid)
        i = ex["intake"]
        flags = []
        for name in ("industry", "budget", "timeline"):
            if i[name]["status"] != "stated":
                flags.append(f"- {name}: {i[name]['status']}. {i[name].get('note') or ''}")
        miss = "\n".join(f"- {m}" for m in i["missing_fields"])
        client.chat_postMessage(channel=channel, text=(
            "*Flagged from the transcript*\n" + ("\n".join(flags) or "- none") +
            "\n\n*Missing information*\n" + miss))
        send_docx(client, channel, sid,
                  "Draft proposal attached. Open items are marked [To confirm: ...]. Ask me a question, "
                  "request a change (e.g. 'make the timeline more detailed'), or say 'approve'.")
    except Exception as e:
        client.chat_postMessage(channel=channel, text=f"Sorry, generation failed: {e}")


def handle_text(client, channel, sid, text):
    db.log_message(sid, "user", text)
    secs = db.current_sections(sid)
    if not secs:
        client.chat_postMessage(channel=channel, text="Upload a client transcript first.")
        return
    r = route(text, secs)
    if r.intent == "question":
        ans = answer_question(sid, text)
        db.log_message(sid, "assistant", ans)
        client.chat_postMessage(channel=channel, text=ans)
    elif r.intent == "revise" and r.section_key:
        client.chat_postMessage(channel=channel, text=f"Updating only the '{r.section_key}' section...")
        revise_section(sid, r.section_key, r.instruction or text)
        send_docx(client, channel, sid, f"Revised DOCX. Only the '{r.section_key}' section changed. "
                                        "Anything else, or approve?")
    elif r.intent == "approve":
        client.chat_postMessage(channel=channel, text="Approved. The final DOCX is the last one I sent.")
    else:
        client.chat_postMessage(channel=channel, text="I can answer questions about the client, revise a "
                                                      "section, or take new transcripts.")


@app.event("message")
def on_message(event, client):
    if event.get("bot_id") or event.get("subtype") not in (None, "file_share"):
        return
    channel = event["channel"]
    sid = channel  # one session per DM/channel: the same session continues the next day
    db.ensure_session(sid)
    files = event.get("files") or []
    if files:
        threading.Thread(target=process_upload, args=(client, channel, sid, files), daemon=True).start()
    elif event.get("text"):
        threading.Thread(target=handle_text, args=(client, channel, sid, event["text"]), daemon=True).start()


if __name__ == "__main__":
    db.init()
    ingest_if_empty()
    threading.Thread(target=lambda: __import__("uvicorn").run(api, port=8000, log_level="warning"),
                     daemon=True).start()
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()