import os
import sys
import streamlit as st

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

from src.agents.graph import create_research_graph
from src.tools.vector_store import LocalVectorStore
from config import DEFAULT_MODEL_NAME, MAX_REVISIONS

# Page Config
st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🤖 Agent Configuration")
st.sidebar.markdown("---")

api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    value=os.getenv("GOOGLE_API_KEY", ""),
    help="Get a free key from Google AI Studio: https://aistudio.google.com/"
)
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

st.sidebar.info(
    f"**Engine Specs:**\n"
    f"- **Orchestration:** LangGraph\n"
    f"- **LLM:** `{DEFAULT_MODEL_NAME}`\n"
    f"- **Max Self-Corrections:** {MAX_REVISIONS}\n"
    f"- **Search:** DuckDuckGo + ChromaDB"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📜 System Architecture")
st.sidebar.caption("1. **Planner Node:** Sub-question decomposition\n2. **Research Node:** Hybrid RAG & Web Search\n3. **Writer Node:** Synthesis with inline citations\n4. **Critic Node:** Groundedness evaluation & dynamic loop")

# Main Title & Hero
st.title("🤖 Agentic Research Assistant")
st.caption("Autonomous Multi-Agent Collaboration Engine powered by LangGraph, Google Gemini, and Hybrid Retrieval.")

# PDF Ingestion Section (Optional Local RAG)
with st.expander("📄 Upload Knowledge Base PDFs (Optional Local Vector Store RAG)", expanded=False):
    uploaded_files = st.file_uploader("Upload PDF or TXT files to query alongside web search", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files and st.button("Ingest Documents into ChromaDB"):
        vector_store = LocalVectorStore()
        texts = []
        metadatas = []
        for file in uploaded_files:
            content = file.read().decode("utf-8", errors="ignore")
            texts.append(content)
            metadatas.append({"source": file.name})
        
        vector_store.add_documents(texts, metadatas)
        st.success(f"Successfully ingested {len(texts)} document(s) into local vector store!")

# Research Query Input
topic = st.text_input(
    "Enter a Research Topic or Prompt:",
    placeholder="e.g., Impact of Quantum Computing on Modern Cybersecurity in 2026",
    value=""
)

col1, col2 = st.columns([1, 4])
with col1:
    start_button = st.button("🚀 Start Multi-Agent Research", type="primary", use_container_width=True)

if start_button:
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Please enter a valid Google Gemini API Key in the sidebar to proceed.")
        st.stop()
        
    if not topic.strip():
        st.warning("Please enter a research topic first.")
        st.stop()

    # Progress Container
    status_container = st.status("🤖 Orchestrating Multi-Agent Workflow...", expanded=True)
    
    # Initialize LangGraph State
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

    # Execute LangGraph Pipeline
    try:
        app_graph = create_research_graph()
        
        # Invoke Graph
        final_state = app_graph.invoke(initial_state)
        
        # Display Step-by-Step Execution Logs
        for log in final_state.get("status_log", []):
            status_container.write(log)
            
        status_container.update(label="✅ Multi-Agent Workflow Completed!", state="complete", expanded=False)

        # Tabs for Results
        tab_report, tab_sources, tab_metrics = st.tabs(["📝 Final Research Report", "🌐 Retrieved Evidence & Sources", "📊 Critic Metrics & State"])
        
        with tab_report:
            st.markdown(final_state.get("final_report", "No report generated."))
            
        with tab_sources:
            st.subheader("Web Search Results")
            for res in final_state.get("web_results", []):
                st.markdown(f"- **[{res.get('title')}]({res.get('url')})**")
                st.caption(res.get('snippet'))
                
            st.subheader("Local Vector Store Documents")
            for doc in final_state.get("retrieved_docs", []):
                st.markdown(f"- **Source:** `{doc.get('source')}`")
                st.caption(doc.get('content'))

        with tab_metrics:
            st.json({
                "Critic Quality Score": f"{final_state.get('critic_score', 0.0):.2f}/1.00",
                "Revision Iterations": final_state.get("revision_count", 0),
                "Sub-questions Analyzed": final_state.get("sub_questions", []),
                "Executed Search Queries": final_state.get("search_queries", []),
                "Critic Feedback": final_state.get("critic_feedback", "")
            })

    except Exception as e:
        status_container.update(label="❌ Error executing agent workflow", state="error", expanded=True)
        st.error(f"Execution Error: {str(e)}")
