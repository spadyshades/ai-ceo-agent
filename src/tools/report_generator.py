"""Generate a PDF executive report from an agent run."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

from src.config import DB_PATH


def _sanitize(text: str) -> str:
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "BMW CEO Strategic Intelligence Report", align="R")
        self.ln(10)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_x(10)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 80)
        self.ln(6)
        self.cell(0, 10, _sanitize(title))
        self.ln(10)

    def body_text(self, text: str):
        self.set_x(10)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, _sanitize(text))
        self.ln(2)

    def label_value(self, label: str, value: str):
        self.set_x(10)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 6, f"{_sanitize(label)}: {_sanitize(value)}")


def generate_report(
    run_id: str | None = None,
    output_path: str | None = None,
) -> str:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if run_id:
        run = conn.execute(
            "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
        ).fetchone()
    else:
        run = conn.execute(
            "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()

    if not run:
        conn.close()
        raise ValueError("No agent runs found")

    run = dict(run)
    state = json.loads(run.get("state_json", "{}"))
    conn.close()

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/report_{ts}.pdf"

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 20, 60)
    pdf.cell(0, 15, "Executive Intelligence Briefing", align="C")
    pdf.ln(12)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")
    pdf.ln(4)
    pdf.cell(0, 6, _sanitize(f"Goal: {run['goal'][:120]}"), align="C")
    pdf.ln(12)

    briefing = state.get("briefing", "")
    if briefing:
        pdf.section_title("CEO Briefing")
        pdf.body_text(briefing)

    recs = state.get("recommendations", [])
    if recs:
        pdf.section_title(f"Strategic Recommendations ({len(recs)})")
        for i, rec in enumerate(recs, 1):
            priority = rec.get("priority", "Medium")
            conf = rec.get("confidence", 0)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 30, 80)
            pdf.multi_cell(0, 7, _sanitize(f"{i}. [{priority}] {rec.get('title', '')} ({conf:.0%})"))
            pdf.ln(2)

            pdf.label_value("Rationale", rec.get("rationale", ""))
            pdf.label_value("Expected Impact", rec.get("expected_impact", ""))
            pdf.label_value("Risk Assessment", rec.get("risk_assessment", ""))

            sources = rec.get("evidence_sources", [])
            evidence = rec.get("evidence_chunk_ids", [])
            pdf.label_value("Evidence", f"{len(evidence)} chunks from {', '.join(sources) if sources else 'n/a'}")
            pdf.ln(4)

    opps = state.get("opportunities", [])
    if opps:
        pdf.section_title(f"Opportunities ({len(opps)})")
        for opp in opps:
            impact = opp.get("impact", "Medium")
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 6, _sanitize(f"[{impact}] {opp.get('title', '')}"))
            pdf.body_text(opp.get("description", ""))

    risks = state.get("risks", [])
    if risks:
        pdf.section_title(f"Risks ({len(risks)})")
        for risk in risks:
            impact = risk.get("impact", "Medium")
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 6, _sanitize(f"[{impact}] {risk.get('title', '')}"))
            pdf.body_text(risk.get("description", ""))

    pdf.section_title("Run Metadata")
    pdf.label_value("Run ID", run["id"])
    pdf.label_value("Status", run.get("status", ""))
    pdf.label_value("Started", run.get("started_at", "")[:19])
    pdf.label_value("Finished", (run.get("finished_at") or "")[:19])
    pdf.label_value("Replans", str(state.get("replan_count", 0)))
    pdf.label_value("Tool Calls", str(len(state.get("tool_results", []))))

    val = state.get("validation")
    if val:
        pdf.label_value("Validation", "PASSED" if val.get("passed") else "FAILED")
        issues = val.get("issues", [])
        if issues:
            for iss in issues:
                pdf.body_text(f"  - {iss}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)
    return output_path


def main() -> None:
    path = generate_report()
    print(f"Report saved to: {path}")


if __name__ == "__main__":
    main()
