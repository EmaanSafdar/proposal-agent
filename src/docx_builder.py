import json
import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(r"(\[To confirm:[^\]]*\])")


def _add_text(par, text: str) -> None:
    for part in MARKER.split(text):
        if not part:
            continue
        run = par.add_run(part)
        if MARKER.fullmatch(part):  # open items shown in red for the consultant
            run.bold = True
            run.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)


def _add_body(doc, body: str) -> None:
    for raw in body.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- "):
            _add_text(doc.add_paragraph(style="List Bullet"), line[2:])
        else:
            _add_text(doc.add_paragraph(), line)


def build_docx(plan_sections: list[dict], texts: dict[str, str], path: Path) -> Path:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    for s in plan_sections:
        body = texts[s["key"]]
        if s["role"] == "cover":
            doc.add_heading(s["title"], 0)
            _add_body(doc, body)
            doc.add_page_break()
        else:
            doc.add_heading(s["title"], 1)
            _add_body(doc, body)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path


if __name__ == "__main__":
    draft = json.loads((ROOT / "data" / "draft_cache.json").read_text(encoding="utf-8"))
    out = build_docx(draft["plan"]["sections"], draft["sections"],
                     ROOT / "data" / "output" / "proposal_test.docx")
    print("Saved", out)