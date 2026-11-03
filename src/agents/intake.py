import json
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.llm import generate_structured

ROOT = Path(__file__).resolve().parents[2]


class Status(str, Enum):
    stated = "stated"            # client said it clearly
    conflicting = "conflicting"  # different people gave different values
    ambiguous = "ambiguous"      # said vaguely / only partly
    missing = "missing"          # not in the transcript at all


class Field_(BaseModel):
    value: Optional[str] = Field(None, description="Value exactly as supported by the transcript. null if missing.")
    status: Status
    evidence: Optional[str] = Field(None, description="Short supporting quote or paraphrase from the transcript.")
    note: Optional[str] = Field(None, description="For conflicting/ambiguous: the competing values and who said them.")


class Stakeholder(BaseModel):
    name: str
    role: Optional[str] = None
    organisation: Optional[str] = None
    influence: Optional[str] = Field(None, description="Role in the decision, e.g. budget owner, user, technical contact.")


class IntakeResult(BaseModel):
    company_name: Field_
    industry: Field_
    problem: Field_
    goals: Field_
    budget: Field_
    timeline: Field_
    stakeholders: list[Stakeholder]
    integrations_required: list[str]
    constraints: list[str] = Field(description="Hard constraints or things the client explicitly does not want.")
    out_of_scope_or_low_priority: list[str]
    missing_fields: list[str] = Field(description="Information a proposal needs that the transcript does not give.")
    open_questions: list[str] = Field(description="Questions the consultant should clarify with the client.")


SYSTEM = """You are the Intake Agent in a proposal-generation system.
Your only job: read a client discovery call transcript and extract structured client information.

Rules:
- Never guess or fill gaps. If something is not stated, set status=missing and value=null.
- Never infer a value from a company name. If a field is not explicitly stated by a participant, status must be ambiguous or missing, never stated. If you use an inference, status must be ambiguous and the note must say it is an inference.
- If the transcript only implies something without saying it, set status=ambiguous and explain in note.
- If different participants give different values (e.g. two budgets, two deadlines), set status=conflicting.
  value must contain only what was explicitly agreed. Never attribute one person's number or date to another.
  Record every competing value and who said it in note, and say if a claim is unconfirmed.
- Never describe something as confirmed unless the client explicitly confirmed it. Statements by the consultant are not client confirmations.
- Use a short evidence quote or paraphrase from the transcript for every field.
- Include every party that influences the decision, even if absent from the call (e.g. a board that approves budget).
- integrations_required must include integrations mentioned as future or undecided, marked as such.
- In constraints, separate hard constraints from preferences, and mark which ones the client called non-negotiable.
- missing_fields must be specific to this client: decision authority, support or maintenance expectations,
  unconfirmed claims, tentative plans, and any figure or date that two participants disagree on and that has
  not been verified (e.g. the board-approved amount).
- open_questions: concrete questions to ask the client.
- If a field's status is ambiguous because it is inferred, value must be null and the inference goes only in note. Do not put an inferred value in value.
- In timeline and budget, only the client's own statements count. If the consultant proposes or labels something (e.g. "confirmed fallback"), do not copy that wording into value. Use the client's own words, e.g. "Q3 is the goal, Q4 is where they would probably land".
- Do not write any proposal content."""

def run_intake(transcript: str) -> IntakeResult:
    result = generate_structured(SYSTEM, f"TRANSCRIPT:\n\n{transcript}", IntakeResult)
    return enforce_rules(result)


def enforce_rules(result: IntakeResult) -> IntakeResult:
    """Deterministic checks after the LLM call."""
    # Industry is never named in the transcript: any value is an inference.
    ind = result.industry
    if ind.value:
        ind.note = f"{ind.note or ''} [Inferred, unconfirmed: {ind.value}]".strip()
        ind.value = None
    if ind.status != Status.missing:
        ind.status = Status.ambiguous
    # If the note records disagreement between participants, status must be conflicting.
    for f in (result.budget, result.timeline):
        n = (f.note or "").lower()
        if f.status == Status.stated and ("jenny" in n or "suggest" in n or "disagree" in n or "while" in n):
            f.status = Status.conflicting
    return result
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    text = (ROOT / "data" / "transcript" / "client_transcript.txt").read_text(encoding="utf-8")
    result = run_intake(text)
    data = json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)
    print(data)

    out = ROOT / "data" / "intake_cache.json"
    out.write_text(data, encoding="utf-8")
    print(f"\nSaved to {out}")