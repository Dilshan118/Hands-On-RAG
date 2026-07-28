import os
import sys
import tempfile
import streamlit as st

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.data_loader import load_multi_source_data
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStoreManager

# Page Config
st.set_page_config(
    page_title="Modular RAG Pipeline Workbench",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }

    .gradient-title {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }

    .stButton > button {
        border-radius: 8px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Config
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.5rem 0 1rem 0;">
    <h2 style="margin: 0; font-size: 1.4rem; color: #ffffff;">📚 MODULAR RAG</h2>
    <span style="font-size: 0.75rem; color: #38bdf8; font-weight: 600;">Vector Retrieval Engine</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ Pipeline Parameters")

provider_choice = st.sidebar.selectbox(
    "Embedding Provider",
    options=["Local HuggingFace (SentenceTransformers)", "Google Gemini Embeddings"],
    index=0
)

if "Local" in provider_choice:
    emb_provider = "local"
    emb_model = st.sidebar.text_input("SentenceTransformer Model", value="all-MiniLM-L6-v2")
else:
    emb_provider = "google"
    emb_model = st.sidebar.text_input("Gemini Embedding Model", value="models/gemini-embedding-001")
    gemini_key = st.sidebar.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key

vector_db_choice = st.sidebar.selectbox(
    "Vector Store Provider",
    options=["ChromaDB (Local Persistent)", "Pinecone (Cloud DB)"],
    index=0
)

vector_db_provider = "chroma" if "Chroma" in vector_db_choice else "pinecone"

# Pinecone Target Dimension Matching
target_dim = None
if vector_db_provider == "pinecone":
    target_dim = st.sidebar.number_input(
        "Pinecone Index Vector Dimension",
        min_value=64,
        max_value=4096,
        value=1024,
        step=128,
        help="Must match your exact Pinecone index dimension (e.g. 1024 for 'elegant-walnut')."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 Chunking Settings")
chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)

# Main Dashboard Header
st.markdown('<h1 class="gradient-title">Modular RAG Ingestion & Retrieval Workbench</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.05rem;'>Interactive pipeline for ingesting multi-source documents, chunking text, generating embeddings, and inspecting vector similarity search.</p>", unsafe_allow_html=True)

tab_ingest, tab_search = st.tabs(["📥 Ingestion & Vector Indexing", "🔍 Semantic Vector Search"])

with tab_ingest:
    st.subheader("Multi-Source Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, CSV, or TXT documents to index into the Vector Store",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("⚡ Process, Chunk & Index Documents", type="primary"):
        pdf_paths = []
        csv_paths = []
        txt_paths = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.read())
                    
                if file.name.endswith(".pdf"):
                    pdf_paths.append(temp_path)
                elif file.name.endswith(".csv"):
                    csv_paths.append(temp_path)
                elif file.name.endswith(".txt"):
                    txt_paths.append(temp_path)

            with st.spinner("Step 1/3: Loading multi-source raw documents..."):
                raw_docs = load_multi_source_data(
                    pdf_files=pdf_paths,
                    csv_files=csv_paths,
                    text_files=txt_paths
                )
                
            if not raw_docs:
                st.error("No valid document content extracted.")
                st.stop()
                
            st.success(f"Loaded {len(raw_docs)} raw document page(s)/row(s).")
            
            with st.spinner(f"Step 2/3: Chunking documents (size={chunk_size}, overlap={chunk_overlap})..."):
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunked_docs = splitter.split_documents(raw_docs)
                
            st.success(f"Generated {len(chunked_docs)} chunk(s).")
            
            dim_info = f" ({target_dim} dims)" if target_dim else ""
            with st.spinner(f"Step 3/3: Generating embeddings{dim_info} & indexing into {vector_db_provider.upper()}..."):
                emb_manager = EmbeddingManager(provider=emb_provider, model_name=emb_model, dimension=target_dim)
                texts = [doc.page_content for doc in chunked_docs]
                embeddings = emb_manager.generate_embeddings(texts)
                
                v_store = VectorStoreManager(provider=vector_db_provider, collection_name="rag_workbench_docs")
                v_store.add_documents(chunked_docs, embeddings)
                
            st.balloons()
            st.success(f"✅ Successfully indexed {len(chunked_docs)} vector embeddings into {vector_db_provider.upper()}!")

with tab_search:
    st.subheader("Execute Semantic Similarity Query")
    query_text = st.text_input("Semantic Query Prompt:", placeholder="e.g., What are the core topics discussed in the document?")
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=3)
    
    if st.button("🔎 Run Similarity Search", type="primary"):
        if not query_text.strip():
            st.warning("Please enter a query first.")
            st.stop()
            
        with st.spinner("Generating query vector & retrieving top-k matches..."):
            try:
                emb_manager = EmbeddingManager(provider=emb_provider, model_name=emb_model, dimension=target_dim)
                query_vec = emb_manager.generate_embedding(query_text)
                
                v_store = VectorStoreManager(provider=vector_db_provider, collection_name="rag_workbench_docs")
                results = v_store.search_similarity(query_vec, top_k=top_k)
                
                if results:
                    st.markdown(f"### Retrieved {len(results)} Relevant Chunk(s):")
                    for i, res in enumerate(results, 1):
                        meta = res.get("metadata", {})
                        source = meta.get("source", "Unknown Source")
                        score = res.get("score")
                        score_str = f" | **Similarity Score:** `{score:.4f}`" if score is not None else ""
                        
                        st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="margin-top: 0; color: #38bdf8;">Result #{i} — Source: <code>{source}</code>{score_str}</h4>
                            <div style="font-size: 0.95rem; line-height: 1.5; color: #e2e8f0; margin-top: 0.5rem;">{res.get('text')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No matching chunks found. Please ensure you have processed and indexed documents in the Ingestion tab.")
            except Exception as e:
                st.error(f"Search Error: {str(e)}")
