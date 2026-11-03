import json
from typing import Literal, Optional

from pydantic import BaseModel

from src import db
from src.agents.intake import IntakeResult
from src.agents.writer import PlanSection, ProposalPlan, write_section
from src.llm import generate_structured, generate_text


class Route(BaseModel):
    intent: Literal["question", "revise", "approve", "other"]
    section_key: Optional[str] = None
    instruction: Optional[str] = None


ROUTER_SYSTEM = """You route a consultant's message about a draft proposal.
- question: asks about the client/transcript/draft content.
- revise: asks to change a part of the proposal. Set section_key to the best matching key from the list, and
  instruction to what must change.
- approve: says the proposal is fine / approved.
- other: anything else.
Only choose a section_key from the provided list."""

ANSWER_SYSTEM = """Answer the consultant's question using ONLY the transcript, intake data and draft provided.
If the answer is not there, say so. Quote what participants said where useful, and attribute it to the speaker."""


def route(message: str, sections: list[dict]) -> Route:
    keys = "\n".join(f"- {s['key']}: {s['title']}" for s in sections)
    return generate_structured(ROUTER_SYSTEM, f"SECTIONS:\n{keys}\n\nMESSAGE: {message}", Route)


def answer_question(sid: str, question: str) -> str:
    ex = db.get_extraction(sid)
    secs = "\n\n".join(f"[{s['title']}]\n{s['text']}" for s in db.current_sections(sid))
    prompt = (f"TRANSCRIPTS:\n{db.all_transcript_text(sid)}\n\nINTAKE:\n{json.dumps(ex['intake'], indent=2)}\n\n"
              f"DRAFT:\n{secs}\n\nQUESTION: {question}")
    return generate_text(ANSWER_SYSTEM, prompt, temperature=0.2)


def revise_section(sid: str, key: str, instruction: str) -> str:
    ex = db.get_extraction(sid)
    intake = IntakeResult.model_validate(ex["intake"])
    plan = ProposalPlan.model_validate(ex["plan"])
    sec = next(s for s in plan.sections if s.key == key)
    sections = db.current_sections(sid)
    current = next(s for s in sections if s["key"] == key)
    others = {s["key"]: s["text"] for s in sections if s["key"] != key}
    feedback = (f"The consultant asked for this change to the existing section. Apply it, keep everything "
                f"else consistent.\nCHANGE REQUEST: {instruction}\n\nCURRENT TEXT:\n{current['text']}")
    new_text = write_section(sec, plan, intake, ex["research"], others, feedback=feedback)
    db.save_section(sid, key, current["title"], current["role"], current["position"], new_text)
    return new_text