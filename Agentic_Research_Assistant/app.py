import os
import sys
import time
import streamlit as st

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

from src.agents.graph import create_research_graph
from src.tools.vector_store import LocalVectorStore
from config import DEFAULT_MODEL_NAME, MAX_REVISIONS

# Page Config
st.set_page_config(
    page_title="Agentic Research Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Card Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }

    /* Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-container:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 0.25rem;
    }
    .metric-lbl {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
    }

    /* Buttons Micro-animations */
    .stButton > button {
        border-radius: 10px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
    }

    /* Gradient Text */
    .gradient-title {
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.75rem;
        font-weight: 800;
    }

    /* Status Pill */
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Header Card
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.5rem 0 1rem 0;">
    <h2 style="margin: 0; font-size: 1.5rem; color: #ffffff;">🤖 AGENTIC ENGINE</h2>
    <span class="status-pill">StateGraph Multi-Agent System</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# LLM Provider Configuration
st.sidebar.markdown("### ⚙️ Engine Settings")

provider = st.sidebar.selectbox(
    "LLM Provider",
    options=["Groq (Recommended Free API)", "Google Gemini", "Ollama (Local Offline LLM)"],
    index=0,
    help="Groq offers ultra-fast, high-quota free inference for Llama-3.3-70B."
)

if "Groq" in provider:
    os.environ["LLM_PROVIDER"] = "groq"
    groq_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get a 100% free key at console.groq.com"
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        
    selected_model = st.sidebar.selectbox(
        "Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0
    )
    os.environ["GROQ_MODEL"] = selected_model

elif "Google" in provider:
    os.environ["LLM_PROVIDER"] = "google"
    gemini_key = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Get a free key at aistudio.google.com"
    )
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
        
    selected_model = st.sidebar.selectbox(
        "Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )
    os.environ["GEMINI_MODEL"] = selected_model

else:
    os.environ["LLM_PROVIDER"] = "ollama"
    selected_model = st.sidebar.selectbox(
        "Ollama Model",
        options=["llama3.2", "mistral", "qwen2.5"],
        index=0,
        help="Ensure 'ollama serve' is running locally."
    )
    os.environ["OLLAMA_MODEL"] = selected_model

st.sidebar.markdown("---")

# Active Engine Specs
st.sidebar.markdown("### 📊 Active Specifications")
st.sidebar.markdown(f"""
<div class="glass-card" style="font-size: 0.85rem; line-height: 1.6;">
    <div><b>Orchestration:</b> <code>LangGraph StateGraph</code></div>
    <div><b>Provider:</b> <code>{os.getenv('LLM_PROVIDER', 'groq').upper()}</code></div>
    <div><b>Active Model:</b> <code>{selected_model}</code></div>
    <div><b>Max Revisions:</b> <code>{MAX_REVISIONS} Loops</code></div>
    <div><b>Retrieval:</b> <code>ChromaDB + Multi-Threaded DDGS</code></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧱 Architecture Topology")
st.sidebar.caption("""
1. **🎯 Planner:** Task Decomposition & Query Cleaning
2. **🌐 Research:** Concurrent Web & Vector Search
3. **✍️ Writer:** Synthesis & Citation Indexing
4. **🧐 Critic:** Groundedness Evaluation & Loop Gate
""")

# Main Content Hero
st.markdown('<h1 class="gradient-title">Agentic Research Assistant</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #8b949e; font-size: 1.1rem; margin-bottom: 2rem;'>Stateful Autonomous Multi-Agent Collaboration Engine with Parallel Retrieval and Self-Correction Loops.</p>", unsafe_allow_html=True)

# Knowledge Base PDF Ingestion
with st.expander("📄 Optional: Upload PDF/TXT Documents (Local Vector Store RAG)", expanded=False):
    uploaded_files = st.file_uploader("Upload files to query alongside live web search", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files and st.button("Ingest Files into ChromaDB"):
        vector_store = LocalVectorStore()
        texts = []
        metadatas = []
        for file in uploaded_files:
            content = file.read().decode("utf-8", errors="ignore")
            texts.append(content)
            metadatas.append({"source": file.name})
        
        vector_store.add_documents(texts, metadatas)
        st.success(f"Successfully ingested {len(texts)} file(s) into local vector store!")

# Research Query Input
topic = st.text_input(
    "Research Topic or Technical Prompt:",
    placeholder="e.g., Impact of Quantum Computing on Modern Cryptography in 2026",
    value=""
)

col_btn, col_blank = st.columns([1, 3])
with col_btn:
    start_button = st.button("🚀 Start Multi-Agent Research", type="primary", use_container_width=True)

if start_button:
    current_provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if current_provider == "groq" and not os.getenv("GROQ_API_KEY"):
        st.error("Please enter a valid Groq API Key in the sidebar to proceed.")
        st.stop()
    elif current_provider == "google" and not os.getenv("GOOGLE_API_KEY"):
        st.error("Please enter a valid Google Gemini API Key in the sidebar to proceed.")
        st.stop()
        
    if not topic.strip():
        st.warning("Please enter a research topic first.")
        st.stop()

    start_time = time.time()

    # Execution Progress Container
    status_container = st.status("🤖 Orchestrating Stateful Agent Graph...", expanded=True)
    
    initial_state = {
        "topic": topic.strip(),
        "sub_questions": [],
        "search_queries": [],
        "retrieved_docs": [],
        "web_results": [],
        "draft_report": "",
        "critic_score": 0.0,
        "critic_feedback": "",
        "revision_count": 0,
        "final_report": "",
        "status_log": []
    }

    try:
        app_graph = create_research_graph()
        final_state = app_graph.invoke(initial_state)
        elapsed_time = time.time() - start_time
        
        for log in final_state.get("status_log", []):
            status_container.write(log)
            
        status_container.update(label=f"✅ Multi-Agent Graph Completed in {elapsed_time:.2f}s!", state="complete", expanded=False)

        st.markdown("<br>", unsafe_allow_html=True)

        # 4-Column Key Metrics Display
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-val">{final_state.get('critic_score', 0.0):.2f}/1.0</div>
                <div class="metric-lbl">Groundedness Score</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-val">{elapsed_time:.2f}s</div>
                <div class="metric-lbl">Execution Latency</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-val">{final_state.get('revision_count', 0)}</div>
                <div class="metric-lbl">Reflection Loops</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-val">{len(final_state.get('web_results', []))}</div>
                <div class="metric-lbl">Web Sources</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Output Navigation Tabs
        tab_report, tab_sources, tab_state = st.tabs(["📝 Final Research Report", "🌐 Evidence & Sources", "📊 Engine Metrics & State"])
        
        with tab_report:
            report_text = final_state.get("final_report", "No report generated.")
            st.markdown(report_text)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Research Report (.md)",
                data=report_text,
                file_name=f"research_report_{topic[:20].strip().replace(' ', '_')}.md",
                mime="text/markdown"
            )
            
        with tab_sources:
            st.subheader("🌐 Live Web Search Evidence")
            web_res = final_state.get("web_results", [])
            if web_res:
                for res in web_res:
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4 style="margin-top: 0;"><a href="{res.get('url')}" target="_blank" style="color: #818cf8; text-decoration: none;">{res.get('title')}</a></h4>
                        <div style="font-size: 0.8rem; color: #8b949e; margin-bottom: 0.5rem;"><b>URL:</b> <code>{res.get('url')}</code></div>
                        <div style="font-size: 0.9rem; line-height: 1.5; color: #c9d1d9;">{res.get('snippet')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No external web results fetched.")
                
            st.subheader("📄 Local Vector Store Context")
            vec_docs = final_state.get("retrieved_docs", [])
            if vec_docs:
                for doc in vec_docs:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div><b>Source Document:</b> <code>{doc.get('source')}</code></div>
                        <div style="font-size: 0.9rem; color: #c9d1d9; margin-top: 0.5rem;">{doc.get('content')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No local vector store documents queried.")

        with tab_state:
            st.json({
                "Execution Latency": f"{elapsed_time:.2f} seconds",
                "Critic Quality Score": f"{final_state.get('critic_score', 0.0):.2f}/1.00",
                "Revision Iterations": final_state.get("revision_count", 0),
                "Sub-questions Decomposed": final_state.get("sub_questions", []),
                "Executed Search Queries": final_state.get("search_queries", []),
                "Critic Review Notes": final_state.get("critic_feedback", "")
            })

    except Exception as e:
        status_container.update(label="❌ Error executing agent graph", state="error", expanded=True)
        st.error(f"Execution Error: {str(e)}")
