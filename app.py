import streamlit as st
import streamlit.components.v1 as components
from streamlit_pdf_viewer import pdf_viewer
import pandas as pd
import plotly.express as px
from backend.ingest_service import IngestService
from backend.query_service import QueryService
from graphrag.utils.logger import logger
from graphrag.utils.config import get_config

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="GraphRAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== LOAD CSS =====================
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===================== SESSION STATE =====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None
if "ragas_metrics" not in st.session_state:
    st.session_state.ragas_metrics = None
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🤖"

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)

    # LLM
    st.markdown('<div class="sidebar-section"><h4>🧠 LLM</h4>', unsafe_allow_html=True)
    llm_provider = st.selectbox("Provider", ["openai", "cohere"], index=0, label_visibility="collapsed")
    model_name = st.text_input("Model", value="gpt-3.5-turbo" if llm_provider=="openai" else "command-r", label_visibility="collapsed")
    api_key = st.text_input("API Key", type="password", placeholder="Enter your API key", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Embedding
    st.markdown('<div class="sidebar-section"><h4>📐 Embedding</h4>', unsafe_allow_html=True)
    embedding_model = st.text_input(
        "Model", 
        value=get_config("EMBEDDING_MODEL", "intfloat/e5-base-v2"), 
        disabled=True, 
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Retrieval Mode
    st.markdown('<div class="sidebar-section"><h4>🔍 Retrieval Mode</h4>', unsafe_allow_html=True)
    retrieval_mode = st.radio("Mode", ["Vector", "Graph", "Hybrid"], index=2, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Top-K
    st.markdown('<div class="sidebar-section"><h4>📊 Top-K</h4>', unsafe_allow_html=True)
    default_top_k = get_config("TOP_K_RETRIEVAL", 5)
    top_k = st.slider("Number of chunks", 1, 10, default_top_k, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Advanced
    st.markdown('<div class="sidebar-section"><h4>⚙️ Advanced</h4>', unsafe_allow_html=True)
    use_triplets = st.checkbox("Include triplets", value=get_config("USE_TRIPLETS", True))
    default_context = get_config("WITH_CONTEXT", False)
    with_context = st.checkbox("Context-aware retrieval", value=default_context)
    default_context_size = get_config("CONTEXT_SIZE", 2)
    context_size = st.slider("Context size", 1, 5, default_context_size, disabled=not with_context)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Upload PDF
    st.markdown('<div class="sidebar-section"><h4>📄 Upload PDF</h4>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")
    if uploaded_file is not None:
        st.session_state.pdf_bytes = uploaded_file.getvalue()
        with st.spinner("Processing PDF..."):
            ingest = IngestService()
            doc_id = uploaded_file.name.split('.')[0]
            max_tokens = get_config("MAX_TOKENS_PER_CHUNK", 200)
            success = ingest.process_pdf(st.session_state.pdf_bytes, doc_id, max_tokens=max_tokens)
            if success:
                st.session_state.doc_id = doc_id
                st.success("✅ PDF ingested successfully!")
                logger.info(f"PDF {doc_id} ingested successfully")
            else:
                st.error("❌ Ingest failed. Check logs.")
                logger.error(f"PDF {doc_id} ingestion failed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Database Status
    st.markdown('<div class="sidebar-section"><h4>Database Status</h4>', unsafe_allow_html=True)
    ingest = IngestService()
    # Kiểm tra Neo4j
    try:
        from graphrag.connectors.neo4j_connection import get_connection as get_neo4j
        neo4j = get_neo4j()
        neo4j_ok = neo4j.test_connection()
    except:
        neo4j_ok = False
    # Kiểm tra Qdrant
    qdrant_ok = ingest.check_qdrant()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="db-status-item"><span class="indicator {"green" if neo4j_ok else "red"}"></span> Neo4j</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="db-status-item"><span class="indicator {"green" if qdrant_ok else "red"}"></span> Qdrant</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.graph_html = None
        st.session_state.ragas_metrics = None
        logger.info("Chat cleared")
        st.rerun()

# ===================== MAIN CHAT AREA =====================
st.title("🤖 GraphRAG Assistant")
st.caption("Ask questions about your documents with GraphRAG")

chat_container = st.container(height=400, border=False)
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                st.markdown(msg["content"])

query = st.chat_input("Type your question here...")
if query and st.session_state.doc_id is not None:
    st.session_state.messages.append({"role": "user", "content": query})
    with chat_container:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(query)

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Thinking..."):
                config = {
                    "llm_type": llm_provider,
                    "model": model_name,
                    "api_key": api_key,
                    "retrieval_mode": retrieval_mode,
                    "top_k": top_k,
                    "use_triplets": use_triplets,
                    "with_context": with_context,
                    "context_size": context_size
                }
                qs = QueryService()
                answer, graph_html, metrics = qs.query(query, config)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.graph_html = graph_html
                st.session_state.ragas_metrics = metrics

# ===================== BOTTOM: PDF Preview | Graph | RAGAS =====================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="bottom-column"><h4>📄 PDF Preview</h4>', unsafe_allow_html=True)
    if st.session_state.pdf_bytes is not None:
        pdf_viewer(input=st.session_state.pdf_bytes, height=350)
    else:
        st.info("No PDF uploaded yet")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="bottom-column"><h4>🌐 Graph Visualization</h4>', unsafe_allow_html=True)
    if st.session_state.graph_html is not None:
        components.html(st.session_state.graph_html, height=350)
    else:
        st.info("Graph will appear here after a query")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="bottom-column"><h4>📊 Retrieval & RAGAS</h4>', unsafe_allow_html=True)
    if st.session_state.ragas_metrics is not None:
        df = pd.DataFrame([st.session_state.ragas_metrics])
        st.dataframe(df, use_container_width=True)
        fig = px.bar(
            df.melt(var_name="Metric", value_name="Score"),
            x="Metric",
            y="Score",
            color="Metric",
            title="RAGAS Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Metrics will appear here after a query")
    st.markdown('</div>', unsafe_allow_html=True)