from src.agents.reviewer import review_section
import json
import sys
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from src.agents.intake import IntakeResult
from src.llm import generate_structured, generate_text

ROOT = Path(__file__).resolve().parents[2]


class Role(str, Enum):
    cover = "cover"
    executive_summary = "executive_summary"
    proposed_solution = "proposed_solution"
    timeline = "timeline"
    budget_range = "budget_range"
    next_steps = "next_steps"
    additional = "additional"


REQUIRED = {Role.cover, Role.executive_summary, Role.proposed_solution,
            Role.timeline, Role.budget_range, Role.next_steps}


class PlanSection(BaseModel):
    key: str = Field(description="snake_case identifier, unique")
    title: str
    role: Role = Field(description="Which required part this fulfils, or 'additional'.")
    purpose: str = Field(description="What this section must achieve for THIS client.")
    intake_facts_to_use: list[str]
    research_to_use: list[str] = Field(description="Which past-proposal patterns/figures are useful here.")


class ProposalPlan(BaseModel):
    reasoning: str = Field(description="Why this structure suits this client.")
    sections: list[PlanSection]


PLAN_SYSTEM = """You are the planning step of the Writer Agent in a proposal-generation system.
Decide the structure of a proposal for this specific client, using the intake data and research notes.

The proposal must cover these six parts (set role accordingly): cover, executive_summary,
proposed_solution, timeline, budget_range, next_steps.
Beyond those, add extra sections only if this client's situation calls for them (for example assumptions,
risks, integrations, open items) and justify them in reasoning. Do not copy a past proposal's
section list mechanically. Order the sections logically for a reader at the client.
For each section state its purpose for this client, which intake facts it must use, and which research
patterns are useful. Plan only; do not write proposal content."""

WRITE_SYSTEM = """You are the Writer Agent. You write ONE section of a client proposal at a time.

Rules:
- Client facts come only from the intake data. Past proposals are references for patterns and indicative
  figures only. Never present another client's facts as this client's, and never name other clients.
- Do not guess. If a fact needed here is missing or unconfirmed, write a visible marker:
  "[To confirm: ...]". Do not invent names, dates, numbers or technical details.
- Where participants disagreed, use the client's own agreed wording and show the open point honestly.
  Do not use the word "confirmed" unless the client confirmed it.
- Budget: when participants gave conflicting figures, anchor the proposal on the figure the client agreed
  as the working target. Do not present another participant's unverified figure as a ceiling, range or
  board-approved amount. Mention the disagreement only as an open point in the budget section using
  "[To confirm: ...]".
- Do not state that a third party (e.g. a board) approved something unless the intake marks it as confirmed.
- Use "non-negotiable" only for items intake marks as non-negotiable. Treat other requirements as client
  preferences, for example "strong preference".
- Low-priority items are not in scope unless stated; mention them as a possible later phase at most.
- Do not promise real-time sync unconditionally. State that real-time depends on what the existing system's
  interfaces allow, to be validated in discovery.
- Anything borrowed from a past proposal (assumptions, support retainers, durations, cost breakdowns) must be
  labelled indicative and framed as a proposal for the client to confirm, never as something the client agreed to.
- Locations: describe only the locations the intake says exist today as current; mention planned locations
  as planned.
- Include a "[To confirm: ...]" marker wherever intake lists a missing or conflicting item relevant to this
  section. Do not put "[To confirm: ...]" markers in the cover section.
- Write for the client reader: clear, professional, specific to their situation. Plain text, short paragraphs,
  simple "- " bullets where useful. No markdown headings, no title line (the title is added separately).
- Output only the section body."""


def _research_context(research: dict, limit: int = 12) -> str:
    chunks = sorted(research.get("chunks", []), key=lambda c: c.get("distance", 1))[:limit]
    parts = [f"RESEARCH NOTES (reference only):\n{research.get('notes', '')}"]
    for c in chunks:
        parts.append(f"--- {c['client']} | {c['industry']} | {c['section_title']}\n{c['text']}")
    return "\n\n".join(parts)


def plan_proposal(intake: IntakeResult, research: dict) -> ProposalPlan:
    prompt = ("INTAKE:\n" + json.dumps(intake.model_dump(mode="json"), indent=2)
              + "\n\n" + _research_context(research, limit=8))
    plan = generate_structured(PLAN_SYSTEM, prompt, ProposalPlan)
    missing = REQUIRED - {s.role for s in plan.sections}
    if missing:  # code-level check of the hard requirement, one retry with feedback
        prompt += f"\n\nYour previous plan lacked these required parts: {sorted(m.value for m in missing)}. Include them."
        plan = generate_structured(PLAN_SYSTEM, prompt, ProposalPlan)
        missing = REQUIRED - {s.role for s in plan.sections}
        if missing:
            raise ValueError(f"Plan still missing required sections: {missing}")
    return plan


def write_section(section: PlanSection, plan: ProposalPlan, intake: IntakeResult,
                  research: dict, written: dict[str, str], feedback: str = "") -> str:
    outline = "\n".join(f"- {s.title}: {s.purpose}" for s in plan.sections)
    earlier = "\n\n".join(f"[{k}]\n{v}" for k, v in written.items()) or "(none yet)"
    prompt = (
        f"PROPOSAL OUTLINE:\n{outline}\n\n"
        f"SECTION TO WRITE NOW:\nTitle: {section.title}\nRole: {section.role.value}\nPurpose: {section.purpose}\n"
        f"Intake facts to use: {section.intake_facts_to_use}\nResearch to use: {section.research_to_use}\n\n"
        f"INTAKE:\n{json.dumps(intake.model_dump(mode='json'), indent=2)}\n\n"
        f"{_research_context(research)}\n\n"
        f"SECTIONS ALREADY WRITTEN (keep consistent, do not repeat):\n{earlier}"
    )
    if feedback:
        prompt += f"\n\nREVIEWER FEEDBACK: fix these problems in your rewrite:\n{feedback}"
    return generate_text(WRITE_SYSTEM, prompt)


def run_writer(intake: IntakeResult, research: dict) -> dict:
    plan = plan_proposal(intake, research)
    written: dict[str, str] = {}
    reviews: dict[str, list] = {}
    for s in plan.sections:
        text = write_section(s, plan, intake, research, written)
        review = review_section(s.title, text, intake)
        if review.issues:  # one rewrite pass using the reviewer's feedback
            fb = "\n".join(f"- '{i.claim}': {i.problem}" for i in review.issues)
            text = write_section(s, plan, intake, research, written, feedback=fb)
            reviews[s.key] = [i.model_dump() for i in review.issues]
        written[s.key] = text
    return {"plan": plan.model_dump(mode="json"), "sections": written, "reviews": reviews}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    intake = IntakeResult.model_validate_json((ROOT / "data" / "intake_cache.json").read_text(encoding="utf-8"))
    research = json.loads((ROOT / "data" / "research_cache.json").read_text(encoding="utf-8"))
    draft = run_writer(intake, research)

    print("PLAN REASONING:\n", draft["plan"]["reasoning"], "\n")
    for s in draft["plan"]["sections"]:
        print(f"=== {s['title']}  [{s['role']}] ===")
        print(draft["sections"][s["key"]], "\n")

    out = ROOT / "data" / "draft_cache.json"
    out.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")