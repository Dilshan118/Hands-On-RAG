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
    page_title="Agentic Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🤖 Agent Configuration")
st.sidebar.markdown("---")

provider = st.sidebar.selectbox(
    "Select LLM Provider",
    options=["Groq (Recommended Free API)", "Google Gemini", "Ollama (Local Offline LLM)"],
    index=0,
    help="Groq offers ultra-fast, high-quota free inference for Llama-3.3-70B with zero 429 quota issues."
)

if "Groq" in provider:
    os.environ["LLM_PROVIDER"] = "groq"
    groq_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get a 100% free key from Groq Console: https://console.groq.com/"
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        
    selected_model = st.sidebar.selectbox(
        "Select Groq Model",
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
        help="Get a free key from Google AI Studio: https://aistudio.google.com/"
    )
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
        
    selected_model = st.sidebar.selectbox(
        "Select Gemini Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )
    os.environ["GEMINI_MODEL"] = selected_model

else:
    os.environ["LLM_PROVIDER"] = "ollama"
    selected_model = st.sidebar.selectbox(
        "Select Ollama Local Model",
        options=["llama3.2", "mistral", "qwen2.5"],
        index=0,
        help="Make sure Ollama is installed and running locally ('ollama serve')."
    )
    os.environ["OLLAMA_MODEL"] = selected_model

st.sidebar.info(
    f"**Engine Specs:**\n"
    f"- **Orchestration:** LangGraph\n"
    f"- **Provider:** `{os.getenv('LLM_PROVIDER', 'groq').upper()}`\n"
    f"- **LLM:** `{selected_model}`\n"
    f"- **Max Self-Corrections:** {MAX_REVISIONS}\n"
    f"- **Search:** DuckDuckGo + ChromaDB"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📜 System Architecture")
st.sidebar.caption("1. **Planner Node:** Sub-question decomposition\n2. **Research Node:** Parallel Web Search & Vector Store\n3. **Writer Node:** Synthesis with inline citations\n4. **Critic Node:** Groundedness evaluation & dynamic loop")

# Main Title & Hero
st.title("🤖 Agentic Research Assistant")
st.caption("Autonomous Multi-Agent Collaboration Engine powered by LangGraph, Groq / Gemini, and Multi-Threaded Retrieval.")

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
        elapsed_time = time.time() - start_time
        
        # Display Step-by-Step Execution Logs
        for log in final_state.get("status_log", []):
            status_container.write(log)
            
        status_container.update(label=f"✅ Multi-Agent Workflow Completed in {elapsed_time:.2f}s!", state="complete", expanded=False)

        # Tabs for Results
        tab_report, tab_sources, tab_metrics = st.tabs(["📝 Final Research Report", "🌐 Retrieved Evidence & Sources", "📊 Critic Metrics & State"])
        
        with tab_report:
            report_text = final_state.get("final_report", "No report generated.")
            st.markdown(report_text)
            
            # Export Download Button
            st.download_button(
                label="📥 Download Research Report (.md)",
                data=report_text,
                file_name=f"research_report_{topic[:20].strip().replace(' ', '_')}.md",
                mime="text/markdown"
            )
            
        with tab_sources:
            st.subheader("🌐 Live Web Search Results")
            web_res = final_state.get("web_results", [])
            if web_res:
                for res in web_res:
                    st.markdown(f"### [{res.get('title')}]({res.get('url')})")
                    st.caption(f"**Source URL:** `{res.get('url')}`")
                    st.write(res.get('snippet'))
                    st.markdown("---")
            else:
                st.info("No web results fetched.")
                
            st.subheader("📄 Local Vector Store Documents")
            vec_docs = final_state.get("retrieved_docs", [])
            if vec_docs:
                for doc in vec_docs:
                    st.markdown(f"- **Source File:** `{doc.get('source')}`")
                    st.caption(doc.get('content'))
            else:
                st.info("No local vector store documents queried.")

        with tab_metrics:
            st.json({
                "Total Execution Time": f"{elapsed_time:.2f} seconds",
                "Critic Quality Score": f"{final_state.get('critic_score', 0.0):.2f}/1.00",
                "Revision Iterations": final_state.get("revision_count", 0),
                "Sub-questions Analyzed": final_state.get("sub_questions", []),
                "Executed Search Queries": final_state.get("search_queries", []),
                "Critic Feedback": final_state.get("critic_feedback", "")
            })

    except Exception as e:
        status_container.update(label="❌ Error executing agent workflow", state="error", expanded=True)
        st.error(f"Execution Error: {str(e)}")
