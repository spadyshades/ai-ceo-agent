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

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460; border-radius: 10px;
        padding: 1.2rem; text-align: center;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; color: #e94560; }
    .metric-card p {
        margin: 0.3rem 0 0 0; color: #a8a8b3;
        font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;
    }
    .section-explain {
        color: #8d8d9b; font-size: 0.88rem; margin-bottom: 1rem;
        border-left: 3px solid #0f3460; padding-left: 0.8rem;
    }
    .impact-high { color: #e94560; font-weight: 700; }
    .impact-medium { color: #f5a623; font-weight: 700; }
    .impact-low { color: #2ecc71; font-weight: 700; }
    .briefing-box {
        background: #16213e; border: 1px solid #0f3460;
        border-radius: 8px; padding: 1.5rem; line-height: 1.7; font-size: 1.05rem;
    }
    .run-status-pass { color: #2ecc71; font-weight: bold; }
    .run-status-fail { color: #e94560; font-weight: bold; }
    div[data-testid="stExpander"] {
        border: 1px solid #1a1a3e; border-radius: 6px; margin-bottom: 0.5rem;
    }
    .fin-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border: 1px solid #1a3a6e; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .fin-card h3 { margin: 0; font-size: 1.4rem; color: #4fc3f7; }
    .fin-card p { margin: 0.2rem 0 0 0; color: #a8a8b3; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# --- data helpers ---

@st.cache_data(ttl=30)
def _load_runs():
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT 20").fetchall()
    conn.close(); return [dict(r) for r in rows]

def _load_state(run):
    raw = run.get("state_json")
    if not raw: return {}
    try: return json.loads(raw)
    except: return {}

@st.cache_data(ttl=30)
def _load_tool_calls(run_id):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM agent_tool_calls WHERE run_id = ? ORDER BY called_at", (run_id,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

@st.cache_data(ttl=30)
def _load_validations(run_id):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM agent_validations WHERE run_id = ? ORDER BY validated_at", (run_id,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

@st.cache_data(ttl=30)
def _load_plans(run_id):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM agent_plans WHERE run_id = ? ORDER BY created_at", (run_id,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

@st.cache_data(ttl=60)
def _load_corpus_stats():
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    total_docs = conn.execute("SELECT COUNT(*) as n FROM documents").fetchone()["n"]
    indexed = conn.execute("SELECT COUNT(DISTINCT document_id) as n FROM chunks").fetchone()["n"]
    sources = conn.execute("SELECT source, COUNT(*) as n FROM documents GROUP BY source ORDER BY n DESC").fetchall()
    recent = conn.execute("SELECT source, title, published_at FROM documents ORDER BY published_at DESC LIMIT 15").fetchall()
    conn.close()
    return {"total_docs": total_docs, "indexed": indexed, "sources": [dict(s) for s in sources], "recent": [dict(r) for r in recent]}

@st.cache_data(ttl=120)
def _load_precomputed_sentiment():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT title, source, published_at, sentiment_label, sentiment_score FROM document_sentiments "
            "WHERE published_at IS NOT NULL AND published_at != '' ORDER BY published_at", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df = df.dropna(subset=["published_at"])
    return df

@st.cache_data(ttl=300)
def _load_financial_snapshot():
    try:
        from src.tools.financial_data import get_snapshot
        return get_snapshot("BMW.DE")
    except Exception:
        return None

@st.cache_data(ttl=300)
def _load_competitor_data():
    try:
        from src.tools.competitor_comparison import compare
        return compare()
    except Exception:
        return []

@st.cache_data(ttl=300)
def _load_topic_trends():
    try:
        from src.tools.topic_trends import detect_rising_topics
        return detect_rising_topics(days_recent=14, days_baseline=60, top_n=15)
    except Exception:
        return []

def _metric_card(value, label):
    return f'<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>'

def _fin_card(value, label):
    return f'<div class="fin-card"><h3>{value}</h3><p>{label}</p></div>'

def _impact_span(impact):
    cls = {"High": "impact-high", "Medium": "impact-medium", "Low": "impact-low"}.get(impact, "")
    return f'<span class="{cls}">{impact}</span>'


# --- agent runner ---

def _run_agent_from_dashboard(goal):
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
    status.success(f"Agent run complete in {elapsed:.0f}s -- {n_recs} recommendations, validation {'PASSED' if val_passed else 'FAILED'}, {state.replan_count} replan(s)")
    _load_runs.clear()
    time.sleep(1)
    st.rerun()


# --- sidebar ---

def _sidebar():
    st.sidebar.markdown("## BMW CEO Intelligence")
    st.sidebar.markdown('<p style="color:#8d8d9b;font-size:0.8rem;">Strategic intelligence powered by Qwen 2.5 7B, LangGraph, and ChromaDB</p>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Run New Analysis**")
    goal_input = st.sidebar.text_area("Strategic goal", placeholder="e.g. What should BMW prioritise in China this quarter?", height=80, key="goal_input")
    run_clicked = st.sidebar.button("Run Agent", use_container_width=True, type="primary")
    st.sidebar.caption("Typical runtime: 5-15 minutes on CPU")
    if run_clicked and goal_input.strip():
        st.session_state["run_goal"] = goal_input.strip()

    st.sidebar.markdown("---")

    # PDF export
    st.sidebar.markdown("**Export**")
    if st.sidebar.button("Download PDF Report", use_container_width=True):
        st.session_state["export_pdf"] = True

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

    selected_idx = st.sidebar.selectbox("Select run", range(len(labels)), format_func=lambda i: labels[i])
    run = runs[selected_idx]
    state = _load_state(run)

    st.sidebar.code(f"ID: {run['id'][:12]}...", language=None)
    status_class = "run-status-pass" if run["status"] == "success" else "run-status-fail"
    st.sidebar.markdown(f'Status: <span class="{status_class}">{run["status"].upper()}</span>', unsafe_allow_html=True)

    replans = state.get("replan_count", 0)
    n_recs = len(state.get("recommendations", []))
    n_tools = len(state.get("tool_results", []))
    val = state.get("validation")
    val_status = "PASSED" if val and val.get("passed") else "FAILED"
    st.sidebar.markdown(f"Replans: **{replans}** | Tools: **{n_tools}** | Recs: **{n_recs}**")
    val_class = "run-status-pass" if val_status == "PASSED" else "run-status-fail"
    st.sidebar.markdown(f'Validation: <span class="{val_class}">{val_status}</span>', unsafe_allow_html=True)

    should_run = "run_goal" in st.session_state
    return run, should_run


# --- sections ---

def _section_overview(corpus, state):
    st.header("Company Overview")
    st.markdown('<div class="section-explain">High-level view of the intelligence corpus and live financial snapshot.</div>', unsafe_allow_html=True)

    # Financial snapshot
    snap = _load_financial_snapshot()
    if snap:
        f1, f2, f3, f4, f5 = st.columns(5)
        price_str = f"{snap.currency} {snap.current_price:.2f}" if snap.current_price else "N/A"
        change_str = f"{snap.day_change_pct:+.2f}%" if snap.day_change_pct is not None else "N/A"
        cap_str = "N/A"
        if snap.market_cap:
            cap_str = f"{snap.market_cap/1e9:.1f}B" if snap.market_cap >= 1e9 else f"{snap.market_cap/1e6:.0f}M"
        pe_str = f"{snap.pe_ratio:.1f}" if snap.pe_ratio else "N/A"
        div_str = f"{snap.dividend_yield:.2f}%" if snap.dividend_yield else "N/A"

        with f1: st.markdown(_fin_card(price_str, "Stock Price"), unsafe_allow_html=True)
        with f2: st.markdown(_fin_card(change_str, "Day Change"), unsafe_allow_html=True)
        with f3: st.markdown(_fin_card(cap_str, "Market Cap"), unsafe_allow_html=True)
        with f4: st.markdown(_fin_card(pe_str, "P/E Ratio"), unsafe_allow_html=True)
        with f5: st.markdown(_fin_card(div_str, "Dividend Yield"), unsafe_allow_html=True)
        st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(_metric_card(corpus["total_docs"], "Total Documents"), unsafe_allow_html=True)
    with c2: st.markdown(_metric_card(corpus["indexed"], "Indexed"), unsafe_allow_html=True)
    with c3: st.markdown(_metric_card(len(corpus["sources"]), "Data Sources"), unsafe_allow_html=True)
    with c4: st.markdown(_metric_card(len(state.get("recommendations", [])), "Recommendations"), unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Documents by Source")
        src_df = pd.DataFrame(corpus["sources"])
        if not src_df.empty:
            fig = px.bar(src_df, x="source", y="n", labels={"source": "Source", "n": "Documents"}, color="source", color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False, height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Source Tier Distribution")
        tier_map = {"bmw_press": "Tier 1 (Primary)", "arxiv": "Tier 1 (Primary)", "google_news": "Tier 2 (News)", "yahoo_finance": "Tier 2 (News)", "hackernews": "Tier 3 (Community)"}
        if not src_df.empty:
            src_df["tier"] = src_df["source"].map(tier_map).fillna("Tier 4 (Unknown)")
            tier_agg = src_df.groupby("tier")["n"].sum().reset_index()
            fig = px.pie(tier_agg, values="n", names="tier", color_discrete_sequence=["#e94560", "#f5a623", "#3498db"])
            fig.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


def _section_market_intelligence(corpus):
    st.header("Market Intelligence")
    st.markdown('<div class="section-explain">The most recent documents ingested into the intelligence corpus.</div>', unsafe_allow_html=True)
    source_filter = st.multiselect("Filter by source", options=sorted({d["source"] for d in corpus["recent"]}), default=None)
    docs = corpus["recent"]
    if source_filter:
        docs = [d for d in docs if d["source"] in source_filter]
    if docs:
        for doc in docs:
            pub = doc.get("published_at", "")[:10]
            st.markdown(f"- **[{doc['source']}]** {doc['title'][:120]} `{pub}`")
    else:
        st.info("No documents match the current filter.")


def _section_opportunities(state):
    st.header("Opportunity Monitor")
    st.markdown('<div class="section-explain">Strategic opportunities identified by the intelligence engine, grounded in corpus evidence.</div>', unsafe_allow_html=True)
    opps = state.get("opportunities", [])
    if not opps:
        st.info("No opportunities detected in this run."); return
    impact_filter = st.multiselect("Filter by impact", ["High", "Medium", "Low"], default=["High", "Medium", "Low"], key="opp_filter")
    for opp in [o for o in opps if o.get("impact", "Medium") in impact_filter]:
        conf = opp.get("confidence", 0)
        st.markdown(f'{_impact_span(opp.get("impact","Medium"))} **{opp["title"]}** <span style="color:#8d8d9b">(confidence: {conf:.0%})</span>', unsafe_allow_html=True)
        st.markdown(f"> {opp['description']}")
        sources = opp.get("evidence_sources", []); chunks = opp.get("evidence_chunk_ids", [])
        st.caption(f"Evidence: {len(chunks)} chunk(s) | Sources: {', '.join(sources) if sources else 'n/a'}")
        st.markdown("")


def _section_risks(state):
    st.header("Risk Monitor")
    st.markdown('<div class="section-explain">Strategic risks detected with negative-sentiment weighting and corpus evidence.</div>', unsafe_allow_html=True)
    risks = state.get("risks", [])
    if not risks:
        st.info("No risks detected in this run."); return
    impact_filter = st.multiselect("Filter by impact", ["High", "Medium", "Low"], default=["High", "Medium", "Low"], key="risk_filter")
    for risk in [r for r in risks if r.get("impact", "Medium") in impact_filter]:
        conf = risk.get("confidence", 0)
        st.markdown(f'{_impact_span(risk.get("impact","Medium"))} **{risk["title"]}** <span style="color:#8d8d9b">(confidence: {conf:.0%})</span>', unsafe_allow_html=True)
        st.markdown(f"> {risk['description']}")
        sources = risk.get("evidence_sources", []); chunks = risk.get("evidence_chunk_ids", [])
        st.caption(f"Evidence: {len(chunks)} chunk(s) | Sources: {', '.join(sources) if sources else 'n/a'}")
        st.markdown("")


def _section_sentiment():
    st.header("Sentiment Analysis")
    st.markdown('<div class="section-explain">Corpus-wide sentiment distribution using FinBERT. Pre-computed during processing for instant rendering.</div>', unsafe_allow_html=True)

    df = _load_precomputed_sentiment()
    if df.empty:
        st.warning("No pre-computed sentiment data. Run: python -m src.processing.sentiment_indexer")
        return

    source_filter = st.multiselect("Filter by source", sorted(df["source"].unique()), default=None, key="sent_source")
    if source_filter:
        df = df[df["source"].isin(source_filter)]

    col1, col2 = st.columns(2)
    with col1:
        dist = df["sentiment_label"].value_counts()
        fig = px.pie(values=dist.values, names=dist.index, title="Sentiment Distribution",
                     color=dist.index, color_discrete_map={"positive": "#2ecc71", "neutral": "#636e72", "negative": "#e94560"})
        fig.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df["date"] = df["published_at"].dt.date
        daily = df.groupby(["date", "sentiment_label"]).size().reset_index(name="count")
        fig = px.area(daily, x="date", y="count", color="sentiment_label", title="Sentiment Over Time",
                      color_discrete_map={"positive": "#2ecc71", "neutral": "#636e72", "negative": "#e94560"})
        fig.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    total = len(df)
    pos_pct = (df["sentiment_label"] == "positive").sum() / total * 100 if total else 0
    neg_pct = (df["sentiment_label"] == "negative").sum() / total * 100 if total else 0
    avg_score = df["sentiment_score"].mean() if total else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents Analyzed", total)
    m2.metric("Positive", f"{pos_pct:.1f}%")
    m3.metric("Negative", f"{neg_pct:.1f}%")
    m4.metric("Avg Confidence", f"{avg_score:.3f}")

    with st.expander("Sample sentiment details"):
        show_df = df[["title", "source", "sentiment_label", "sentiment_score"]].head(50).copy()
        show_df["sentiment_score"] = show_df["sentiment_score"].round(3)
        st.dataframe(show_df, use_container_width=True, hide_index=True)


def _section_competitors():
    st.header("Competitor Analysis")
    st.markdown('<div class="section-explain">Mention frequency and FinBERT sentiment comparison across BMW competitors in the corpus.</div>', unsafe_allow_html=True)

    profiles = _load_competitor_data()
    if not profiles:
        st.info("No competitor data available."); return

    # Bar chart of mentions
    comp_df = pd.DataFrame([{"Competitor": p.name, "Mentions": p.mention_count, "Sentiment": p.avg_sentiment_score} for p in profiles])
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(comp_df, x="Competitor", y="Mentions", color="Competitor", title="Corpus Mentions by Competitor",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(comp_df, x="Competitor", y="Sentiment", color="Competitor", title="Average Sentiment by Competitor",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # Detail cards
    for p in profiles:
        with st.expander(f"{p.name} -- {p.mention_count} mentions"):
            s1, s2, s3 = st.columns(3)
            s1.metric("Positive", p.sentiment_positive)
            s2.metric("Neutral", p.sentiment_neutral)
            s3.metric("Negative", p.sentiment_negative)
            if p.sample_titles:
                st.markdown("**Sample mentions:**")
                for t in p.sample_titles[:5]:
                    st.markdown(f"- {t[:120]}")


def _section_topic_trends():
    st.header("Topic Trends")
    st.markdown('<div class="section-explain">TF-IDF based detection of rising topics across time windows. Complements entity-based trend detection by finding thematic shifts.</div>', unsafe_allow_html=True)

    topics = _load_topic_trends()
    if not topics:
        st.info("No rising topics detected."); return

    topic_df = pd.DataFrame([{
        "Topic": t.term,
        "Recent TF-IDF": t.recent_tfidf,
        "Baseline TF-IDF": t.baseline_tfidf,
        "Growth": t.growth_rate if t.growth_rate != float("inf") else 999.0,
        "Growth Label": "new" if t.growth_rate == float("inf") else f"{t.growth_rate:.1f}x",
    } for t in topics])

    fig = px.bar(topic_df.head(15), x="Topic", y="Growth", color="Recent TF-IDF",
                 title="Top Rising Topics (by growth rate)",
                 color_continuous_scale="Reds", hover_data=["Growth Label", "Recent TF-IDF", "Baseline TF-IDF"])
    fig.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(topic_df[["Topic", "Recent TF-IDF", "Baseline TF-IDF", "Growth Label"]],
                 use_container_width=True, hide_index=True)


def _section_recommendations(state):
    st.header("Strategic Recommendations")
    st.markdown('<div class="section-explain">Ranked recommendations validated against the corpus with adversarial challenge and evidence verification.</div>', unsafe_allow_html=True)
    recs = state.get("recommendations", [])
    if not recs:
        st.info("No recommendations in this run."); return

    priorities = [r.get("priority", "Medium") for r in recs]
    p_counts = {p: priorities.count(p) for p in ["High", "Medium", "Low"] if priorities.count(p)}
    cols = st.columns(len(p_counts) + 1)
    cols[0].markdown(_metric_card(len(recs), "Total Recommendations"), unsafe_allow_html=True)
    for idx, (pri, cnt) in enumerate(p_counts.items(), 1):
        cols[idx].markdown(_metric_card(cnt, f"{pri} Priority"), unsafe_allow_html=True)
    st.markdown("")

    for i, rec in enumerate(recs, 1):
        priority = rec.get("priority", "Medium"); conf = rec.get("confidence", 0)
        with st.expander(f"#{i} | {priority} | {rec['title']} | confidence: {conf:.0%}", expanded=(i <= 2)):
            st.markdown(f"**Rationale:** {rec.get('rationale', '')}")
            st.markdown(f"**Expected Impact:** {rec.get('expected_impact', '')}")
            st.markdown(f"**Risk Assessment:** {rec.get('risk_assessment', '')}")
            evidence = rec.get("evidence_chunk_ids", []); sources = rec.get("evidence_sources", [])
            e1, e2 = st.columns(2)
            e1.metric("Evidence Chunks", len(evidence))
            e2.metric("Distinct Sources", len(set(sources)))
            if sources: st.caption(f"Sources: {', '.join(sources)}")
            if evidence: st.caption(f"Chunk IDs: {', '.join(evidence[:8])}")


def _section_briefing(state):
    st.header("CEO Briefing")
    st.markdown('<div class="section-explain">Executive summary: What happened? Why does it matter? What should management do next?</div>', unsafe_allow_html=True)
    briefing = state.get("briefing", "")
    if not briefing:
        st.info("No briefing generated in this run."); return
    st.markdown(f'<div class="briefing-box">{briefing}</div>', unsafe_allow_html=True)


def _section_agent_trace(run, state):
    st.header("Agent Trace")
    st.markdown('<div class="section-explain">Full execution trace: planning, tool calls, validation, and autonomous re-planning decisions.</div>', unsafe_allow_html=True)

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
                if reasoning: st.markdown(f"**Reasoning:** {reasoning}")
                for step in plan_data.get("steps", []):
                    st.markdown(f"  {step.get('id', '?')}. `{step.get('tool', '?')}` -- {step.get('description', '')}")

    if tool_calls:
        st.subheader(f"Tool Calls ({len(tool_calls)})")
        tc_df = pd.DataFrame(tool_calls)
        if not tc_df.empty and "tool" in tc_df.columns:
            tool_counts = tc_df["tool"].value_counts().reset_index()
            tool_counts.columns = ["tool", "calls"]
            fig = px.bar(tool_counts, x="tool", y="calls", color="tool", color_discrete_sequence=px.colors.qualitative.Set2, title="Tool Usage Distribution")
            fig.update_layout(showlegend=False, height=250, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        tc_table = [{"Step": tc.get("step_id","?"), "Tool": tc.get("tool","?"), "Summary": tc.get("summary","")[:120], "Error": tc.get("error","") or "", "Time": tc.get("called_at","")[:19]} for tc in tool_calls]
        st.dataframe(pd.DataFrame(tc_table), use_container_width=True, hide_index=True)

    if validations:
        st.subheader(f"Validation Passes ({len(validations)})")
        for v in validations:
            passed = bool(v.get("passed", 0)); status = "PASSED" if passed else "FAILED"; color = "green" if passed else "red"
            issues = json.loads(v.get("issues_json", "[]"))
            with st.expander(f":{color}[{status}] at {v['validated_at'][:19]}"):
                if issues:
                    for iss in issues: st.markdown(f"- {iss}")
                else: st.markdown("All checks passed.")

    with st.expander("Raw state JSON"):
        st.json({"replan_count": state.get("replan_count",0), "tool_results": len(state.get("tool_results",[])), "retrieved_chunks": len(state.get("retrieved_chunks",[])), "opportunities": len(state.get("opportunities",[])), "risks": len(state.get("risks",[])), "trends": len(state.get("trends",[])), "recommendations": len(state.get("recommendations",[])), "errors": state.get("errors",[])})


# --- main ---

def main():
    run, should_run = _sidebar()

    st.title("BMW CEO Intelligence Dashboard")

    # Handle PDF export
    if st.session_state.get("export_pdf"):
        del st.session_state["export_pdf"]
        try:
            from src.tools.report_generator import generate_report
            run_id = run["id"] if run else None
            path = generate_report(run_id=run_id)
            with open(path, "rb") as f:
                st.download_button("Save PDF Report", f.read(), file_name="bmw_intelligence_report.pdf", mime="application/pdf")
            st.success(f"Report generated: {path}")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

    # Handle agent run
    if should_run and "run_goal" in st.session_state:
        goal = st.session_state.pop("run_goal")
        st.markdown("---")
        _run_agent_from_dashboard(goal)
        return

    if run is None:
        st.warning("No agent runs found. Type a goal in the sidebar and click **Run Agent**.")
        return

    state = _load_state(run)
    corpus = _load_corpus_stats()

    st.markdown(f'<div class="section-explain">Viewing results for: <strong>{run["goal"]}</strong></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "Overview", "Market Intelligence", "Competitors", "Topic Trends",
        "Opportunities", "Risks", "Sentiment",
        "Recommendations", "CEO Briefing", "Agent Trace",
    ])

    with tabs[0]: _section_overview(corpus, state)
    with tabs[1]: _section_market_intelligence(corpus)
    with tabs[2]: _section_competitors()
    with tabs[3]: _section_topic_trends()
    with tabs[4]: _section_opportunities(state)
    with tabs[5]: _section_risks(state)
    with tabs[6]: _section_sentiment()
    with tabs[7]: _section_recommendations(state)
    with tabs[8]: _section_briefing(state)
    with tabs[9]: _section_agent_trace(run, state)


if __name__ == "__main__":
    main()
