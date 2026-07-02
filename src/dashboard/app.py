"""AI CEO Strategic Intelligence Dashboard.

Launch: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import sqlite3
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DB_PATH

st.set_page_config(
    page_title="BMW CEO Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- custom styling ---

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2rem;
        color: #e94560;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        color: #a8a8b3;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-explain {
        color: #8d8d9b;
        font-size: 0.88rem;
        margin-bottom: 1rem;
        border-left: 3px solid #0f3460;
        padding-left: 0.8rem;
    }
    .impact-high { color: #e94560; font-weight: 700; }
    .impact-medium { color: #f5a623; font-weight: 700; }
    .impact-low { color: #2ecc71; font-weight: 700; }
    .briefing-box {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 1.5rem;
        line-height: 1.7;
        font-size: 1.05rem;
    }
    .run-status-pass { color: #2ecc71; font-weight: bold; }
    .run-status-fail { color: #e94560; font-weight: bold; }
    div[data-testid="stExpander"] {
        border: 1px solid #1a1a3e;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .agent-input-box {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- data helpers ---

@st.cache_data(ttl=30)
def _load_runs() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _load_state(run: dict) -> dict:
    raw = run.get("state_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


@st.cache_data(ttl=30)
def _load_tool_calls(run_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM agent_tool_calls WHERE run_id = ? ORDER BY called_at",
        (run_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@st.cache_data(ttl=30)
def _load_validations(run_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM agent_validations WHERE run_id = ? ORDER BY validated_at",
        (run_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@st.cache_data(ttl=30)
def _load_plans(run_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM agent_plans WHERE run_id = ? ORDER BY created_at",
        (run_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@st.cache_data(ttl=60)
def _load_corpus_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    total_docs = conn.execute("SELECT COUNT(*) as n FROM documents").fetchone()["n"]
    indexed = conn.execute(
        "SELECT COUNT(DISTINCT document_id) as n FROM chunks"
    ).fetchone()["n"]
    sources = conn.execute(
        "SELECT source, COUNT(*) as n FROM documents GROUP BY source ORDER BY n DESC"
    ).fetchall()
    recent = conn.execute(
        "SELECT source, title, published_at FROM documents "
        "ORDER BY published_at DESC LIMIT 15"
    ).fetchall()
    conn.close()
    return {
        "total_docs": total_docs,
        "indexed": indexed,
        "sources": [dict(s) for s in sources],
        "recent": [dict(r) for r in recent],
    }


@st.cache_data(ttl=60)
def _load_sentiment_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT published_at, source, title FROM documents "
        "WHERE published_at IS NOT NULL ORDER BY published_at",
        conn,
    )
    conn.close()
    if df.empty:
        return df
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at"])
    return df


def _run_sentiment_on_sample(texts: list[str]) -> list[dict]:
    from src.tools.sentiment import classify_batch
    results = classify_batch(texts)
    return [{"label": r.label, "score": r.score} for r in results]


def _metric_card(value, label: str) -> str:
    return f'<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>'


def _impact_span(impact: str) -> str:
    cls = {"High": "impact-high", "Medium": "impact-medium", "Low": "impact-low"}.get(
        impact, ""
    )
    return f'<span class="{cls}">{impact}</span>'


# --- agent runner ---

def _run_agent_from_dashboard(goal: str) -> None:
    """Run the agent and show live progress in the dashboard."""
    from src.agent.graph import run_agent

    status = st.empty()
    progress = st.progress(0, text="Starting agent...")

    status.info(f"Running agent with goal: {goal}")
    progress.progress(10, text="Planning...")

    started = time.time()
    state = run_agent(goal)
    elapsed = time.time() - started

    progress.progress(100, text="Complete")

    n_recs = len(state.recommendations) if state.recommendations else 0
    val_passed = state.validation.passed if state.validation else False
    val_text = "PASSED" if val_passed else "FAILED"

    status.success(
        f"Agent run complete in {elapsed:.0f}s -- "
        f"{n_recs} recommendations, validation {val_text}, "
        f"{state.replan_count} replan(s)"
    )

    # Clear caches so the new run appears in the dropdown
    _load_runs.clear()
    time.sleep(1)
    st.rerun()


# --- sidebar ---

def _sidebar():
    st.sidebar.markdown("## BMW CEO Intelligence")
    st.sidebar.markdown(
        '<p style="color:#8d8d9b;font-size:0.8rem;">'
        "Strategic intelligence powered by Phi-4 Mini, LangGraph, and ChromaDB</p>",
        unsafe_allow_html=True,
    )

    # --- Agent Runner ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Run New Analysis**")
    goal_input = st.sidebar.text_area(
        "Strategic goal",
        placeholder="e.g. What should BMW prioritise in China this quarter?",
        height=80,
        key="goal_input",
    )
    run_clicked = st.sidebar.button(
        "Run Agent",
        use_container_width=True,
        type="primary",
    )
    st.sidebar.caption("Typical runtime: 5-15 minutes on CPU")

    if run_clicked and goal_input.strip():
        st.session_state["run_goal"] = goal_input.strip()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Previous Runs**")

    runs = _load_runs()
    if not runs:
        st.sidebar.warning("No agent runs found.")
        return None, False

    labels = []
    for r in runs:
        ts = r["started_at"][:16] if r.get("started_at") else "?"
        status = r.get("status", "?")
        goal_short = r.get("goal", "")[:40]
        labels.append(f"{ts} | {status} | {goal_short}")

    selected_idx = st.sidebar.selectbox(
        "Select run",
        range(len(labels)),
        format_func=lambda i: labels[i],
    )

    run = runs[selected_idx]
    state = _load_state(run)

    st.sidebar.code(f"ID: {run['id'][:12]}...", language=None)

    status_class = "run-status-pass" if run["status"] == "success" else "run-status-fail"
    st.sidebar.markdown(
        f'Status: <span class="{status_class}">{run["status"].upper()}</span>',
        unsafe_allow_html=True,
    )

    replans = state.get("replan_count", 0)
    n_recs = len(state.get("recommendations", []))
    n_tools = len(state.get("tool_results", []))
    val = state.get("validation")
    val_status = "PASSED" if val and val.get("passed") else "FAILED"

    st.sidebar.markdown(f"Replans: **{replans}** | Tools: **{n_tools}** | Recs: **{n_recs}**")
    val_class = "run-status-pass" if val_status == "PASSED" else "run-status-fail"
    st.sidebar.markdown(
        f'Validation: <span class="{val_class}">{val_status}</span>',
        unsafe_allow_html=True,
    )

    should_run = "run_goal" in st.session_state
    return run, should_run


# --- sections ---

def _section_overview(corpus: dict, state: dict):
    st.header("Company Overview")
    st.markdown(
        '<div class="section-explain">'
        "High-level view of the intelligence corpus: how many documents were collected, "
        "from which sources, and how they distribute across the pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_metric_card(corpus["total_docs"], "Total Documents"), unsafe_allow_html=True)
    with c2:
        st.markdown(_metric_card(corpus["indexed"], "Indexed"), unsafe_allow_html=True)
    with c3:
        st.markdown(_metric_card(len(corpus["sources"]), "Data Sources"), unsafe_allow_html=True)
    with c4:
        n_recs = len(state.get("recommendations", []))
        st.markdown(_metric_card(n_recs, "Recommendations"), unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Documents by Source")
        src_df = pd.DataFrame(corpus["sources"])
        if not src_df.empty:
            fig = px.bar(
                src_df, x="source", y="n",
                labels={"source": "Source", "n": "Documents"},
                color="source",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                showlegend=False, height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Source Tier Distribution")
        tier_map = {
            "bmw_press": "Tier 1 (Primary)",
            "arxiv": "Tier 1 (Primary)",
            "google_news": "Tier 2 (News)",
            "yahoo_finance": "Tier 2 (News)",
            "hackernews": "Tier 3 (Community)",
        }
        if not src_df.empty:
            src_df["tier"] = src_df["source"].map(tier_map).fillna("Tier 4 (Unknown)")
            tier_agg = src_df.groupby("tier")["n"].sum().reset_index()
            fig = px.pie(
                tier_agg, values="n", names="tier",
                color_discrete_sequence=["#e94560", "#f5a623", "#3498db"],
            )
            fig.update_layout(
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)


def _section_market_intelligence(corpus: dict):
    st.header("Market Intelligence")
    st.markdown(
        '<div class="section-explain">'
        "The most recent documents ingested into the intelligence corpus. "
        "These feed the retriever, sentiment classifier, and trend detector."
        "</div>",
        unsafe_allow_html=True,
    )

    source_filter = st.multiselect(
        "Filter by source",
        options=sorted({d["source"] for d in corpus["recent"]}),
        default=None,
    )

    docs = corpus["recent"]
    if source_filter:
        docs = [d for d in docs if d["source"] in source_filter]

    if docs:
        for doc in docs:
            pub = doc.get("published_at", "")[:10]
            st.markdown(
                f"- **[{doc['source']}]** {doc['title'][:120]} `{pub}`"
            )
    else:
        st.info("No documents match the current filter.")


def _section_opportunities(state: dict):
    st.header("Opportunity Monitor")
    st.markdown(
        '<div class="section-explain">'
        "Strategic opportunities identified by the intelligence engine. Each is grounded "
        "in corpus evidence with impact rating and confidence score."
        "</div>",
        unsafe_allow_html=True,
    )
    opps = state.get("opportunities", [])
    if not opps:
        st.info("No opportunities detected in this run.")
        return

    impact_filter = st.multiselect(
        "Filter by impact", ["High", "Medium", "Low"], default=["High", "Medium", "Low"],
        key="opp_filter",
    )
    filtered = [o for o in opps if o.get("impact", "Medium") in impact_filter]

    for opp in filtered:
        impact = opp.get("impact", "Medium")
        conf = opp.get("confidence", 0)
        st.markdown(
            f'{_impact_span(impact)} **{opp["title"]}** '
            f'<span style="color:#8d8d9b">(confidence: {conf:.0%})</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"> {opp['description']}")
        sources = opp.get("evidence_sources", [])
        chunks = opp.get("evidence_chunk_ids", [])
        st.caption(
            f"Evidence: {len(chunks)} chunk(s) | Sources: {', '.join(sources) if sources else 'n/a'}"
        )
        st.markdown("")


def _section_risks(state: dict):
    st.header("Risk Monitor")
    st.markdown(
        '<div class="section-explain">'
        "Strategic risks detected by the intelligence engine. Risks are prioritised using "
        "negative-sentiment weighting and corpus evidence."
        "</div>",
        unsafe_allow_html=True,
    )
    risks = state.get("risks", [])
    if not risks:
        st.info("No risks detected in this run.")
        return

    impact_filter = st.multiselect(
        "Filter by impact", ["High", "Medium", "Low"], default=["High", "Medium", "Low"],
        key="risk_filter",
    )
    filtered = [r for r in risks if r.get("impact", "Medium") in impact_filter]

    for risk in filtered:
        impact = risk.get("impact", "Medium")
        conf = risk.get("confidence", 0)
        st.markdown(
            f'{_impact_span(impact)} **{risk["title"]}** '
            f'<span style="color:#8d8d9b">(confidence: {conf:.0%})</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"> {risk['description']}")
        sources = risk.get("evidence_sources", [])
        chunks = risk.get("evidence_chunk_ids", [])
        st.caption(
            f"Evidence: {len(chunks)} chunk(s) | Sources: {', '.join(sources) if sources else 'n/a'}"
        )
        st.markdown("")


def _section_sentiment():
    st.header("Sentiment Analysis")
    st.markdown(
        '<div class="section-explain">'
        "Corpus-wide sentiment distribution and time-series trend. Classified using "
        "cardiffnlp/twitter-roberta-base-sentiment-latest on document titles."
        "</div>",
        unsafe_allow_html=True,
    )
    df = _load_sentiment_data()
    if df.empty:
        st.info("No timestamped documents available for sentiment analysis.")
        return

    sample_size = min(100, len(df))
    sample = df.sample(n=sample_size, random_state=42).sort_values("published_at")

    if "sentiment_cache" not in st.session_state:
        with st.spinner("Running sentiment classification on corpus sample..."):
            texts = sample["title"].tolist()
            results = _run_sentiment_on_sample(texts)
            st.session_state["sentiment_cache"] = results

    results = st.session_state["sentiment_cache"]
    sample = sample.reset_index(drop=True)
    sample["sentiment"] = [r["label"] for r in results]
    sample["score"] = [r["score"] for r in results]

    col1, col2 = st.columns(2)
    with col1:
        dist = sample["sentiment"].value_counts()
        fig = px.pie(
            values=dist.values, names=dist.index,
            title="Sentiment Distribution",
            color=dist.index,
            color_discrete_map={
                "positive": "#2ecc71",
                "neutral": "#636e72",
                "negative": "#e94560",
            },
        )
        fig.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sample["date"] = sample["published_at"].dt.date
        daily = sample.groupby(["date", "sentiment"]).size().reset_index(name="count")
        fig = px.area(
            daily, x="date", y="count", color="sentiment",
            title="Sentiment Over Time",
            color_discrete_map={
                "positive": "#2ecc71",
                "neutral": "#636e72",
                "negative": "#e94560",
            },
        )
        fig.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Sample sentiment details"):
        show_df = sample[["title", "source", "sentiment", "score"]].copy()
        show_df["score"] = show_df["score"].round(3)
        st.dataframe(show_df, use_container_width=True, hide_index=True)


def _section_recommendations(state: dict):
    st.header("Strategic Recommendations")
    st.markdown(
        '<div class="section-explain">'
        "Ranked recommendations produced by the agent. Each is grounded in retrieved "
        "evidence, validated against the corpus, and scored for confidence. The adversarial "
        "validator challenges each claim against contradicting evidence before acceptance."
        "</div>",
        unsafe_allow_html=True,
    )
    recs = state.get("recommendations", [])
    if not recs:
        st.info("No recommendations in this run.")
        return

    priorities = [r.get("priority", "Medium") for r in recs]
    p_counts = {p: priorities.count(p) for p in ["High", "Medium", "Low"] if priorities.count(p)}
    cols = st.columns(len(p_counts) + 1)
    cols[0].markdown(
        _metric_card(len(recs), "Total Recommendations"),
        unsafe_allow_html=True,
    )
    for idx, (pri, cnt) in enumerate(p_counts.items(), 1):
        cols[idx].markdown(
            _metric_card(cnt, f"{pri} Priority"),
            unsafe_allow_html=True,
        )

    st.markdown("")

    for i, rec in enumerate(recs, 1):
        priority = rec.get("priority", "Medium")
        conf = rec.get("confidence", 0)
        with st.expander(
            f"#{i} | {priority} | {rec['title']} | confidence: {conf:.0%}",
            expanded=(i <= 2),
        ):
            st.markdown(f"**Rationale:** {rec.get('rationale', '')}")
            st.markdown(f"**Expected Impact:** {rec.get('expected_impact', '')}")
            st.markdown(f"**Risk Assessment:** {rec.get('risk_assessment', '')}")

            evidence = rec.get("evidence_chunk_ids", [])
            sources = rec.get("evidence_sources", [])

            e1, e2 = st.columns(2)
            e1.metric("Evidence Chunks", len(evidence))
            e2.metric("Distinct Sources", len(set(sources)))

            if sources:
                st.caption(f"Sources: {', '.join(sources)}")
            if evidence:
                st.caption(f"Chunk IDs: {', '.join(evidence[:8])}")


def _section_briefing(state: dict):
    st.header("CEO Briefing")
    st.markdown(
        '<div class="section-explain">'
        "Executive summary answering three questions: What happened? Why does it matter? "
        "What should management do next? Generated by the agent after validation."
        "</div>",
        unsafe_allow_html=True,
    )
    briefing = state.get("briefing", "")
    if not briefing:
        st.info("No briefing generated in this run.")
        return
    st.markdown(
        f'<div class="briefing-box">{briefing}</div>',
        unsafe_allow_html=True,
    )


def _section_agent_trace(run: dict, state: dict):
    st.header("Agent Trace")
    st.markdown(
        '<div class="section-explain">'
        "Full execution trace showing the agent\'s planning decisions, tool invocations, "
        "and validation results. This is the audit trail that demonstrates autonomous "
        "multi-step reasoning with conditional re-planning."
        "</div>",
        unsafe_allow_html=True,
    )

    run_id = run["id"]

    tool_calls = _load_tool_calls(run_id)
    validations = _load_validations(run_id)
    plans = _load_plans(run_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_metric_card(len(plans), "Plan Versions"), unsafe_allow_html=True)
    c2.markdown(_metric_card(len(tool_calls), "Tool Calls"), unsafe_allow_html=True)
    c3.markdown(_metric_card(len(validations), "Validations"), unsafe_allow_html=True)
    errors = sum(1 for tc in tool_calls if tc.get("error"))
    c4.markdown(_metric_card(errors, "Tool Errors"), unsafe_allow_html=True)

    st.markdown("")

    if plans:
        st.subheader(f"Plans ({len(plans)} version(s))")
        for p in plans:
            plan_data = json.loads(p.get("plan_json", "{}"))
            replan_idx = p.get("replan_index", 0)
            with st.expander(f"Plan v{replan_idx}", expanded=(replan_idx == 0)):
                reasoning = plan_data.get("reasoning", "")
                if reasoning:
                    st.markdown(f"**Reasoning:** {reasoning}")
                steps = plan_data.get("steps", [])
                for step in steps:
                    st.markdown(
                        f"  {step.get('id', '?')}. `{step.get('tool', '?')}` "
                        f"-- {step.get('description', '')}"
                    )

    if tool_calls:
        st.subheader(f"Tool Calls ({len(tool_calls)})")

        tc_df = pd.DataFrame(tool_calls)
        if not tc_df.empty and "tool" in tc_df.columns:
            tool_counts = tc_df["tool"].value_counts().reset_index()
            tool_counts.columns = ["tool", "calls"]
            fig = px.bar(
                tool_counts, x="tool", y="calls",
                color="tool",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Tool Usage Distribution",
            )
            fig.update_layout(
                showlegend=False, height=250,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        tc_table = []
        for tc in tool_calls:
            tc_table.append({
                "Step": tc.get("step_id", "?"),
                "Tool": tc.get("tool", "?"),
                "Summary": tc.get("summary", "")[:120],
                "Error": tc.get("error", "") or "",
                "Time": tc.get("called_at", "")[:19],
            })
        st.dataframe(pd.DataFrame(tc_table), use_container_width=True, hide_index=True)

    if validations:
        st.subheader(f"Validation Passes ({len(validations)})")
        for v in validations:
            passed = bool(v.get("passed", 0))
            status = "PASSED" if passed else "FAILED"
            color = "green" if passed else "red"
            issues = json.loads(v.get("issues_json", "[]"))
            with st.expander(f":{color}[{status}] at {v['validated_at'][:19]}"):
                if issues:
                    for iss in issues:
                        st.markdown(f"- {iss}")
                else:
                    st.markdown("All checks passed.")

    with st.expander("Raw state JSON"):
        summary = {
            "replan_count": state.get("replan_count", 0),
            "tool_results": len(state.get("tool_results", [])),
            "retrieved_chunks": len(state.get("retrieved_chunks", [])),
            "opportunities": len(state.get("opportunities", [])),
            "risks": len(state.get("risks", [])),
            "trends": len(state.get("trends", [])),
            "recommendations": len(state.get("recommendations", [])),
            "errors": state.get("errors", []),
        }
        st.json(summary)


# --- main ---

def main():
    run, should_run = _sidebar()

    st.title("BMW CEO Intelligence Dashboard")

    # Handle agent run request
    if should_run and "run_goal" in st.session_state:
        goal = st.session_state.pop("run_goal")
        st.markdown("---")
        _run_agent_from_dashboard(goal)
        return

    if run is None:
        st.warning(
            "No agent runs found. Type a goal in the sidebar and click **Run Agent**, "
            "or run from the terminal:\n\n"
            "```\npython -m src.agent.cli\n```"
        )
        return

    state = _load_state(run)
    corpus = _load_corpus_stats()

    # Show goal of selected run
    st.markdown(
        f'<div class="section-explain">Viewing results for: <strong>{run["goal"]}</strong></div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "Overview",
        "Market Intelligence",
        "Opportunities",
        "Risks",
        "Sentiment",
        "Recommendations",
        "CEO Briefing",
        "Agent Trace",
    ])

    with tabs[0]:
        _section_overview(corpus, state)
    with tabs[1]:
        _section_market_intelligence(corpus)
    with tabs[2]:
        _section_opportunities(state)
    with tabs[3]:
        _section_risks(state)
    with tabs[4]:
        _section_sentiment()
    with tabs[5]:
        _section_recommendations(state)
    with tabs[6]:
        _section_briefing(state)
    with tabs[7]:
        _section_agent_trace(run, state)


if __name__ == "__main__":
    main()
