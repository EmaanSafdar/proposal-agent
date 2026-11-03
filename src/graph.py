from typing import TypedDict

from langgraph.graph import END, StateGraph

from src import db
from src.agents.intake import IntakeResult, run_intake
from src.agents.research import run_research
from src.agents.writer import run_writer


class State(TypedDict, total=False):
    session_id: str
    transcript: str
    intake: dict
    research: dict
    draft: dict


def intake_node(s: State) -> State:
    return {"intake": run_intake(s["transcript"]).model_dump(mode="json")}


def research_node(s: State) -> State:
    return {"research": run_research(IntakeResult.model_validate(s["intake"]))}


def writer_node(s: State) -> State:
    draft = run_writer(IntakeResult.model_validate(s["intake"]), s["research"])
    sid = s["session_id"]
    db.save_extraction(sid, s["intake"], s["research"], draft["plan"])
    for i, sec in enumerate(draft["plan"]["sections"]):
        db.save_section(sid, sec["key"], sec["title"], sec["role"], i, draft["sections"][sec["key"]])
    return {"draft": draft}


def build_graph():
    g = StateGraph(State)
    g.add_node("intake", intake_node)
    g.add_node("research", research_node)
    g.add_node("writer", writer_node)
    g.set_entry_point("intake")
    g.add_edge("intake", "research")
    g.add_edge("research", "writer")
    g.add_edge("writer", END)
    return g.compile()


def generate_proposal(session_id: str) -> None:
    build_graph().invoke({"session_id": session_id, "transcript": db.all_transcript_text(session_id)})