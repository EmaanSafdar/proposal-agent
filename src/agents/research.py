import json
import sys
from pathlib import Path

from google.genai import types

from src.agents.intake import IntakeResult
from src.llm import MODEL, client
from src.rag.store import ingest_if_empty, search

ROOT = Path(__file__).resolve().parents[2]

SEARCH_TOOL = types.FunctionDeclaration(
    name="search_past_proposals",
    description=(
        "Semantic search over past proposals. Returns the most relevant sections. "
        "Optionally filter by industry and/or section types."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(type=types.Type.STRING, description="Natural-language description of what to find."),
            "industry": types.Schema(
                type=types.Type.STRING,
                description="Optional. One of: logistics, retail, healthcare, manufacturing, fintech, construction, distribution, consulting, fleet.",
            ),
            "section_types": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="Optional. Any of: cover, summary, solution, timeline, budget, timeline_budget, next_steps, integrations, deliverables, other.",
            ),
            "k": types.Schema(type=types.Type.INTEGER, description="Number of results, default 4."),
        },
        required=["query"],
    ),
)

SYSTEM = """You are the Research Agent in a proposal-generation system.
Your only job: find the most relevant context in the past-proposal knowledge base for the client described by the intake data.
You do not write the proposal.

Use the search_past_proposals tool several times with different angles, for example:
the client's core problem, their constraints (e.g. keep existing systems), required integrations,
how past proposals structured timeline and budget for similar scope, and risks or caveats they flagged.
Decide queries and filters yourself. Use an industry filter only when it helps; cross-industry matches
on the same problem are valuable too.
The client's industry is not confirmed in the intake data, so do not rely on a single industry filter.
When searching timeline or budget patterns, use section_types ["budget", "timeline_budget"] and k of at least 8.
When finished, reply with short research notes: which past proposals are most relevant and why,
what patterns (phasing, pricing, caveats) are worth reusing, and where past proposals differ from this client
(e.g. budget size). Do not invent facts that are not in the retrieved results."""


def run_research(intake: IntakeResult, max_rounds: int = 6) -> dict:
    ingest_if_empty()
    prompt = "CLIENT INTAKE DATA:\n" + json.dumps(intake.model_dump(mode="json"), indent=2)
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=[types.Tool(function_declarations=[SEARCH_TOOL])],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )

    queries, chunks, notes = [], {}, ""
    for _ in range(max_rounds):
        res = client().models.generate_content(model=MODEL, contents=contents, config=config)
        content = res.candidates[0].content
        calls = [p.function_call for p in content.parts if p.function_call]
        if not calls:
            notes = res.text or ""
            break
        contents.append(content)
        responses = []
        for fc in calls:
            a = dict(fc.args)
            hits = search(
                a["query"],
                k=int(a.get("k", 4)),
                industry=a.get("industry"),
                section_type=list(a["section_types"]) if a.get("section_types") else None,
            )
            queries.append({k: v for k, v in a.items() if k != "k"})
            for h in hits:
                chunks[(h["source_file"], h["section_title"])] = h
            responses.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"results": [
                        {"client": h["client"], "industry": h["industry"],
                         "section": h["section_title"], "text": h["text"]}
                        for h in hits
                    ]},
                )
            )
        contents.append(types.Content(role="user", parts=responses))

    return {"queries": queries, "chunks": list(chunks.values()), "notes": notes}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cache = ROOT / "data" / "intake_cache.json"
    intake = IntakeResult.model_validate_json(cache.read_text(encoding="utf-8"))
    result = run_research(intake)
    print("QUERIES THE AGENT CHOSE:")
    for q in result["queries"]:
        print(" ", q)
    print("\nRETRIEVED SECTIONS:")
    for c in result["chunks"]:
        print(f"  {c['distance']:.3f} | {c['industry']:<13} | {c['section_type']:<16} | {c['client']}")
    print("\nRESEARCH NOTES:\n", result["notes"])
    out = ROOT / "data" / "research_cache.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nSaved to {out}")