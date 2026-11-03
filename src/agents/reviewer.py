import json

from pydantic import BaseModel, Field

from src.agents.intake import IntakeResult
from src.llm import generate_structured


class Issue(BaseModel):
    claim: str = Field(description="The exact problematic statement.")
    problem: str = Field(description="Why it is not supported by the intake data.")


class ReviewResult(BaseModel):
    issues: list[Issue]


REVIEW_SYSTEM = """You are the Reviewer Agent. Your only job is to fact-check ONE proposal section against the intake data.
Flag a statement if it:
- presents a figure, range, approval or date as agreed/approved when intake marks it conflicting, unconfirmed or missing;
- attributes one participant's unverified claim to the client or a third party (e.g. a board);
- treats a claim as verified when the intake shows it was only asserted (e.g. an email or document mentioned but not shown);
- uses a date from the call itself (e.g. the call date) as a project date;
- calls something non-negotiable that intake does not mark non-negotiable;
- states client facts that are not in the intake.
Indicative figures labelled as based on comparable projects are allowed. Do not flag style.
If the section is fine, return an empty issues list. Do not rewrite anything."""


def review_section(title: str, text: str, intake: IntakeResult) -> ReviewResult:
    prompt = (f"INTAKE:\n{json.dumps(intake.model_dump(mode='json'), indent=2)}\n\n"
              f"SECTION: {title}\n{text}")
    return generate_structured(REVIEW_SYSTEM, prompt, ReviewResult)