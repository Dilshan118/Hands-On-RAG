import os
import sys
import time
import streamlit as st

sys.path.append(os.path.dirname(__file__))

from src.agents.graph import create_research_graph
from src.tools.vector_store import LocalVectorStore
from config import DEFAULT_MODEL_NAME, MAX_REVISIONS

st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── DESIGN SYSTEM ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

  /* ── GLOBAL RESET ── */
  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", "Inter", sans-serif;
    background-color: #000000 !important;
    color: #f5f5f7;
    -webkit-font-smoothing: antialiased;
  }

  h1, h2, h3, h4 {
    letter-spacing: -0.03em;
    color: #f5f5f7;
  }

  /* ── HIDE STREAMLIT CHROME ── */
  header[data-testid="stHeader"],
  [data-testid="stToolbar"],
  footer { display: none !important; }

  /* ── SIDEBAR COLLAPSE ── */
  [data-testid="stSidebar"] { display: none !important; }
  [data-testid="collapsedControl"] { display: none !important; }

  /* ── PAGE WIDTH & SPACING ── */
  .block-container {
    padding: 0 !important;
    max-width: 100% !important;
  }

  /* ── NAV BAR ── */
  .nav-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 3rem;
    height: 52px;
    background: rgba(0, 0, 0, 0.72);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-bottom: 1px solid #2d2d2f;
  }
  .nav-logo {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f5f5f7;
  }
  .nav-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #86868b;
  }
  .nav-dot { width: 8px; height: 8px; border-radius: 50%; background: #34c759; display: inline-block; }

  /* ── HERO SECTION ── */
  .hero {
    padding: 5rem 2rem 3rem;
    text-align: center;
    max-width: 860px;
    margin: 0 auto;
  }
  .hero-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #0071e3;
    margin-bottom: 1rem;
  }
  .hero-title {
    font-size: clamp(2.5rem, 6vw, 4.25rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.04em;
    color: #f5f5f7;
    margin-bottom: 1.25rem;
  }
  .hero-sub {
    font-size: 1.15rem;
    line-height: 1.6;
    color: #86868b;
    font-weight: 400;
    max-width: 640px;
    margin: 0 auto;
  }

  /* ── PILL TAGS ── */
  .tag-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 1.75rem 0 2.5rem;
  }
  .tag {
    padding: 0.35rem 0.9rem;
    background: #1d1d1f;
    border: 1px solid #3a3a3c;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 500;
    color: #a1a1a6;
    cursor: default;
  }

  /* ── SEARCH AREA ── */
  .search-wrapper {
    max-width: 780px;
    margin: 0 auto 1rem;
  }
  .stTextArea textarea {
    background: #1d1d1f !important;
    border: 1px solid #3a3a3c !important;
    border-radius: 14px !important;
    color: #f5f5f7 !important;
    font-size: 1rem !important;
    padding: 1rem 1.2rem !important;
    line-height: 1.5 !important;
    resize: none !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  .stTextArea textarea:focus {
    border-color: #0071e3 !important;
    box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.18) !important;
  }
  .stTextArea label { color: #86868b !important; font-size: 0.85rem !important; font-weight: 500 !important; }

  /* ── BUTTONS ── */
  .stButton > button {
    border-radius: 9999px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    border: none !important;
  }
  .stButton > button[kind="primary"] {
    background: #0071e3 !important;
    color: #fff !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: #0077ed !important;
    box-shadow: 0 4px 16px rgba(0, 113, 227, 0.45) !important;
    transform: scale(1.02) !important;
  }
  .stButton > button[kind="secondary"] {
    background: #1d1d1f !important;
    border: 1px solid #3a3a3c !important;
    color: #f5f5f7 !important;
  }
  .stButton > button[kind="secondary"]:hover {
    border-color: #0071e3 !important;
    color: #0071e3 !important;
  }

  /* ── CONFIG STRIP ── */
  .config-strip {
    max-width: 780px;
    margin: 0 auto 3rem;
    background: #1d1d1f;
    border: 1px solid #2d2d2f;
    border-radius: 14px;
    padding: 0.25rem 0.5rem;
  }
  .stSelectbox > div > div {
    background-color: transparent !important;
    border: none !important;
    color: #f5f5f7 !important;
    font-size: 0.9rem !important;
  }
  .stTextInput input {
    background: #1d1d1f !important;
    border: 1px solid #3a3a3c !important;
    border-radius: 10px !important;
    color: #f5f5f7 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 0.9rem !important;
  }
  .stTextInput input:focus {
    border-color: #0071e3 !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.18) !important;
  }

  /* ── SECTION DIVIDER ── */
  .section-divider {
    border: none;
    border-top: 1px solid #2d2d2f;
    max-width: 1060px;
    margin: 2rem auto;
  }

  /* ── RESULTS WRAPPER ── */
  .results-wrapper {
    max-width: 1060px;
    margin: 0 auto;
    padding: 0 2rem 4rem;
  }

  /* ── METRICS GRID ── */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2.5rem;
  }
  .metric-card {
    background: #1d1d1f;
    border: 1px solid #2d2d2f;
    border-radius: 14px;
    padding: 1.25rem 1rem;
    text-align: center;
    transition: border-color 0.2s ease;
  }
  .metric-card:hover { border-color: #0071e3; }
  .metric-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem;
    font-weight: 700;
    color: #f5f5f7;
  }
  .metric-sub { font-size: 0.7rem; color: #86868b; }
  .metric-lbl {
    font-size: 0.75rem;
    font-weight: 500;
    color: #6e6e73;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
  }

  /* ── CARDS ── */
  .apple-card {
    background: #1d1d1f;
    border: 1px solid #2d2d2f;
    border-radius: 18px;
    padding: 1.75rem;
    margin-bottom: 1rem;
  }
  .apple-card h4 { margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600; }

  /* ── AGENT CARDS (IDLE STATE) ── */
  .agent-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    max-width: 1060px;
    margin: 2.5rem auto 4rem;
    padding: 0 2rem;
  }
  .agent-card {
    background: #1d1d1f;
    border: 1px solid #2d2d2f;
    border-radius: 18px;
    padding: 1.5rem 1.25rem;
    transition: all 0.2s ease;
  }
  .agent-card:hover { border-color: #0071e3; transform: translateY(-2px); }
  .agent-icon { font-size: 1.75rem; margin-bottom: 0.6rem; }
  .agent-step { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #0071e3; margin-bottom: 0.3rem; }
  .agent-name { font-size: 1rem; font-weight: 700; color: #f5f5f7; margin-bottom: 0.3rem; }
  .agent-desc { font-size: 0.82rem; color: #6e6e73; line-height: 1.4; }

  /* ── TABS ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #1d1d1f;
    border: 1px solid #2d2d2f;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    margin-bottom: 1.5rem;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #86868b !important;
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 1.25rem !important;
  }
  .stTabs [aria-selected="true"] {
    background: #2c2c2e !important;
    color: #f5f5f7 !important;
    font-weight: 600 !important;
  }

  /* ── EXPANDERS ── */
  [data-testid="stExpander"] {
    background: #1d1d1f !important;
    border: 1px solid #2d2d2f !important;
    border-radius: 12px !important;
  }
  [data-testid="stExpander"] summary {
    color: #f5f5f7 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
  }

  /* ── CODE ── */
  code { color: #0071e3 !important; font-size: 0.85em !important; }
</style>
""", unsafe_allow_html=True)

# ─── NAV BAR ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">⚡ Agentic Research Engine</div>
  <div class="nav-badge">
    <span class="nav-dot"></span>
    LangGraph Swarm Active
  </div>
</div>
""", unsafe_allow_html=True)

# ─── HERO SECTION ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-label">Multi-Agent Intelligence Platform</div>
  <div class="hero-title">Research. Think. Verify.</div>
  <div class="hero-sub">Autonomous multi-agent swarm that plans, searches, synthesises, and self-corrects — delivering grounded research reports at scale.</div>
</div>
""", unsafe_allow_html=True)

# ─── FEATURE TAGS ───────────────────────────────────────────────────────────
st.markdown("""
<div class="tag-row">
  <span class="tag">LangGraph StateGraph</span>
  <span class="tag">Parallel Web Retrieval</span>
  <span class="tag">ChromaDB RAG</span>
  <span class="tag">Self-Correcting Loops</span>
  <span class="tag">Citation Synthesis</span>
</div>
""", unsafe_allow_html=True)

# ─── MAIN SEARCH INPUT ──────────────────────────────────────────────────────
_, center_col, _ = st.columns([0.1, 0.8, 0.1])
with center_col:
    # Quick presets
    st.markdown("<p style='font-size:0.82rem; color:#6e6e73; font-weight:500; margin-bottom:0.5rem;'>QUICK PRESETS</p>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    if "topic_val" not in st.session_state:
        st.session_state["topic_val"] = ""

    if p1.button("⚛️ Quantum + Crypto", type="secondary", use_container_width=True):
        st.session_state["topic_val"] = "Impact of Quantum Computing on Modern Cryptography in 2026"
    if p2.button("🤖 DeepSeek vs Llama", type="secondary", use_container_width=True):
        st.session_state["topic_val"] = "DeepSeek-V3 vs Llama-3.3-70B Performance Benchmarks 2025"
    if p3.button("🧬 AI Drug Discovery", type="secondary", use_container_width=True):
        st.session_state["topic_val"] = "AI and Machine Learning in Drug Discovery and Pharmaceutical Research 2025"
    if p4.button("🌍 Climate AI Models", type="secondary", use_container_width=True):
        st.session_state["topic_val"] = "AI Climate Modelling and Carbon Emission Prediction Advances 2025"

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    topic = st.text_area(
        label="Research Prompt",
        placeholder="Ask anything… a research topic, technical question, or scientific inquiry.",
        value=st.session_state["topic_val"],
        height=100,
        label_visibility="collapsed"
    )

    # Config & Action Row
    c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1])
    with c1:
        provider = st.selectbox(
            "Provider",
            options=["Groq (Free API)", "Google Gemini", "Ollama (Local)"],
            index=0,
            label_visibility="collapsed"
        )

    if "Groq" in provider:
        os.environ["LLM_PROVIDER"] = "groq"
        with c2:
            selected_model = st.selectbox(
                "Model",
                options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"],
                index=0,
                label_visibility="collapsed"
            )
        os.environ["GROQ_MODEL"] = selected_model
        with c3:
            groq_key = st.text_input("API Key", type="password", value=os.getenv("GROQ_API_KEY", ""), placeholder="Groq API Key", label_visibility="collapsed")
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key
    elif "Google" in provider:
        os.environ["LLM_PROVIDER"] = "google"
        with c2:
            selected_model = st.selectbox("Model", options=["gemini-1.5-flash", "gemini-1.5-pro"], index=0, label_visibility="collapsed")
        os.environ["GEMINI_MODEL"] = selected_model
        with c3:
            gemini_key = st.text_input("API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""), placeholder="Gemini API Key", label_visibility="collapsed")
            if gemini_key:
                os.environ["GOOGLE_API_KEY"] = gemini_key
    else:
        os.environ["LLM_PROVIDER"] = "ollama"
        with c2:
            selected_model = st.selectbox("Model", options=["llama3.2", "mistral", "qwen2.5"], index=0, label_visibility="collapsed")
        os.environ["OLLAMA_MODEL"] = selected_model
        with c3:
            st.caption("Ollama must be running locally")

    with c4:
        start_button = st.button("Research ↗", type="primary", use_container_width=True)

    # Optional RAG Expander
    with st.expander("📄 Add Context Documents (Optional RAG)", expanded=False):
        uploaded_files = st.file_uploader("Upload .pdf or .txt reference files", type=["pdf", "txt"], accept_multiple_files=True)
        if uploaded_files and st.button("Ingest into ChromaDB"):
            vs = LocalVectorStore()
            texts, metas = [], []
            for f in uploaded_files:
                texts.append(f.read().decode("utf-8", errors="ignore"))
                metas.append({"source": f.name})
            vs.add_documents(texts, metas)
            st.success(f"Ingested {len(texts)} file(s) ✓")

# ─── EXECUTION LOGIC ─────────────────────────────────────────────────────────
if start_button:
    p = os.getenv("LLM_PROVIDER", "groq").lower()
    if p == "groq" and not os.getenv("GROQ_API_KEY"):
        st.error("Please enter your Groq API Key in the field above.")
        st.stop()
    if p == "google" and not os.getenv("GOOGLE_API_KEY"):
        st.error("Please enter your Google Gemini API Key.")
        st.stop()
    if not topic.strip():
        st.warning("Please enter a research topic.")
        st.stop()

    t0 = time.time()
    with st.spinner(""):
        status = st.status("🤖  Orchestrating multi-agent swarm…", expanded=True)

    state = {
        "topic": topic.strip(), "sub_questions": [], "search_queries": [],
        "retrieved_docs": [], "web_results": [], "draft_report": "",
        "critic_score": 0.0, "critic_feedback": "", "revision_count": 0,
        "final_report": "", "status_log": []
    }
    try:
        graph = create_research_graph()
        result = graph.invoke(state)
        elapsed = time.time() - t0
        for log in result.get("status_log", []):
            status.write(log)
        status.update(label=f"✅  Completed in {elapsed:.1f}s", state="complete", expanded=False)
        st.session_state["result"] = result
        st.session_state["elapsed"] = elapsed
        st.session_state["last_topic"] = topic.strip()
    except Exception as e:
        status.update(label="❌  Error", state="error", expanded=True)
        st.error(str(e))
        st.stop()

# ─── RESULTS AREA ───────────────────────────────────────────────────────────
if "result" in st.session_state:
    R = st.session_state["result"]
    T = st.session_state.get("elapsed", 0)
    last_topic = st.session_state.get("last_topic", "")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Metrics Row
    _, mc, _ = st.columns([0.1, 0.8, 0.1])
    with mc:
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"""
        <div class="metric-card">
          <div class="metric-num">{R.get('critic_score',0.0):.2f}<span class="metric-sub">/1.0</span></div>
          <div class="metric-lbl">Groundedness</div>
        </div>""", unsafe_allow_html=True)
        m2.markdown(f"""
        <div class="metric-card">
          <div class="metric-num">{T:.1f}s</div>
          <div class="metric-lbl">Execution Time</div>
        </div>""", unsafe_allow_html=True)
        m3.markdown(f"""
        <div class="metric-card">
          <div class="metric-num">{R.get('revision_count',0)}</div>
          <div class="metric-lbl">Reflection Loops</div>
        </div>""", unsafe_allow_html=True)
        m4.markdown(f"""
        <div class="metric-card">
          <div class="metric-num">{len(R.get('web_results',[]))}</div>
          <div class="metric-lbl">Web Sources</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # Output Tabs
        tab_rep, tab_src, tab_dive, tab_raw = st.tabs(["📝  Research Report", "🌐  Sources", "🔬  Deep Dive", "📊  Swarm Metrics"])

        with tab_rep:
            report = R.get("final_report", "No report generated.")
            st.markdown(report)
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.download_button("↓  Download Report (.md)", data=report,
                               file_name=f"report_{last_topic[:25].replace(' ','_')}.md",
                               mime="text/markdown")

        with tab_src:
            web = R.get("web_results", [])
            if web:
                for res in web:
                    st.markdown(f"""
                    <div class="apple-card">
                      <h4><a href="{res.get('url')}" target="_blank" style="color:#0071e3;text-decoration:none;">{res.get('title','Untitled')}</a></h4>
                      <div style="font-size:0.78rem;color:#6e6e73;margin-bottom:0.5rem;"><code>{res.get('url','')}</code></div>
                      <div style="font-size:0.9rem;color:#a1a1a6;line-height:1.5;">{res.get('snippet','')}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No web results available.")

            vd = R.get("retrieved_docs", [])
            if vd:
                st.subheader("📄 Document Context")
                for d in vd:
                    st.markdown(f"""
                    <div class="apple-card">
                      <code style='font-size:0.85rem;'>{d.get('source','Unknown')}</code>
                      <p style='color:#a1a1a6;font-size:0.9rem;margin-top:0.5rem;line-height:1.5;'>{d.get('content','')}</p>
                    </div>""", unsafe_allow_html=True)

        with tab_dive:
            st.markdown("""
            <div class="apple-card">
              <div class="agent-step" style="margin-bottom:0.4rem;">Claim Verification Mode</div>
              <h3 style="margin:0 0 0.5rem 0;">Deep Dive Investigation</h3>
              <p style="color:#6e6e73;font-size:0.9rem;">Paste any statement from the report to run an isolated agentic verification loop specifically on that claim.</p>
            </div>""", unsafe_allow_html=True)

            dd_claim = st.text_area(
                "Claim to verify",
                placeholder='"Quantum key distribution guarantees information-theoretic security…"',
                height=90, key="dd_input", label_visibility="collapsed"
            )
            if st.button("🔬  Verify this Claim", type="secondary"):
                if not dd_claim.strip():
                    st.warning("Paste a claim first.")
                else:
                    dd_topic = f"Investigate and verify: \"{dd_claim.strip()}\""
                    dd_status = st.status("Running deep dive agent loop…", expanded=True)
                    dd_state = {
                        "topic": dd_topic, "sub_questions": [], "search_queries": [],
                        "retrieved_docs": [], "web_results": [], "draft_report": "",
                        "critic_score": 0.0, "critic_feedback": "", "revision_count": 0,
                        "final_report": "", "status_log": []
                    }
                    try:
                        dd_t0 = time.time()
                        dd_graph = create_research_graph()
                        dd_result = dd_graph.invoke(dd_state)
                        dd_elapsed = time.time() - dd_t0
                        for log in dd_result.get("status_log", []):
                            dd_status.write(log)
                        dd_status.update(label=f"✅  Deep Dive done in {dd_elapsed:.1f}s", state="complete", expanded=False)
                        st.session_state["dd_result"] = dd_result
                        st.session_state["dd_elapsed"] = dd_elapsed
                        st.session_state["dd_claim"] = dd_claim.strip()
                    except Exception as e:
                        dd_status.update(label="❌ Error", state="error")
                        st.error(str(e))

            if "dd_result" in st.session_state:
                dr = st.session_state["dd_result"]
                de = st.session_state.get("dd_elapsed", 0)
                dc = st.session_state.get("dd_claim", "")

                dc1, dc2, dc3 = st.columns(3)
                dc1.markdown(f"""<div class="metric-card"><div class="metric-num">{dr.get('critic_score',0):.2f}</div><div class="metric-lbl">Score</div></div>""", unsafe_allow_html=True)
                dc2.markdown(f"""<div class="metric-card"><div class="metric-num">{de:.1f}s</div><div class="metric-lbl">Latency</div></div>""", unsafe_allow_html=True)
                dc3.markdown(f"""<div class="metric-card"><div class="metric-num">{len(dr.get('web_results',[]))}</div><div class="metric-lbl">Sources</div></div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                st.markdown(dr.get("final_report", ""))
                st.download_button("↓  Download Deep Dive (.md)",
                                   data=dr.get("final_report", ""),
                                   file_name=f"deepdive_{dc[:25].replace(' ','_')}.md",
                                   mime="text/markdown", key="dd_dl")

        with tab_raw:
            st.json({
                "Execution Latency (s)": round(T, 3),
                "Critic Score": round(R.get("critic_score", 0.0), 4),
                "Revision Count": R.get("revision_count", 0),
                "Sub-questions": R.get("sub_questions", []),
                "Search Queries": R.get("search_queries", []),
                "Critic Feedback": R.get("critic_feedback", ""),
            })

# ─── IDLE STATE ─────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class="agent-grid">
      <div class="agent-card">
        <div class="agent-icon">🎯</div>
        <div class="agent-step">Step 01</div>
        <div class="agent-name">Planner</div>
        <div class="agent-desc">Decomposes complex topics into precise sub-questions and search queries.</div>
      </div>
      <div class="agent-card">
        <div class="agent-icon">🌐</div>
        <div class="agent-step">Step 02</div>
        <div class="agent-name">Researcher</div>
        <div class="agent-desc">Executes parallel web search and ChromaDB vector retrieval concurrently.</div>
      </div>
      <div class="agent-card">
        <div class="agent-icon">✍️</div>
        <div class="agent-step">Step 03</div>
        <div class="agent-name">Writer</div>
        <div class="agent-desc">Synthesises all evidence into a structured, well-cited research report.</div>
      </div>
      <div class="agent-card">
        <div class="agent-icon">🧐</div>
        <div class="agent-step">Step 04</div>
        <div class="agent-name">Critic</div>
        <div class="agent-desc">Scores groundedness and gates the loop — revising until quality passes.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
