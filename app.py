import streamlit as st
from src.rag_pipeline import RAGPipeline
import warnings
warnings.filterwarnings("ignore")

# Load pipeline once
@st.cache_resource
def load_pipeline():
    return RAGPipeline(
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama3.2",
        use_hybrid=True,
        use_reranking=True
    )

rag = load_pipeline()

st.title("RAG")
st.write("Ask any questions about Hr,Finance,It policy.")

# ========== User Options ==========
chunk_strategy = st.selectbox(
    "Select Chunking Strategy",
    options=["recursive", "semantic"]
)

use_multi_query = st.checkbox("Use multi-query", value=True)
use_compression = st.checkbox("Compress context", value=False)
top_k = st.slider("Top K documents to retrieve", min_value=1, max_value=10, value=5)
rerank_top_n = st.slider("Top N documents to rerank", min_value=1, max_value=5, value=3)

# Build / Refresh Index
if st.button("Build / Refresh Index"):
    documents_path = r"C:\Users\evilk\OneDrive\Desktop\Projects\RAG-Complete-Pipeline\data"
    index_save_path = r"C:\Users\evilk\OneDrive\Desktop\Projects\RAG-Complete-Pipeline\chroma_db"
    
    with st.spinner("Building index..."):
        rag.build_index(
            documents_path=documents_path,
            chunk_strategy=chunk_strategy,
            save_path=index_save_path
        )
    st.success(f"Index built with {chunk_strategy} chunking!")

# Question input
question = st.text_input("Enter your question:")

# Fetch answer
if st.button("Get Answer") and question:
    with st.spinner("Fetching answer..."):
        response = rag.query(
            question=question,
            use_multi_query=use_multi_query,
            use_compression=use_compression,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            verbose=False
        )

    st.subheader("Answer")
    st.write(response['answer'])

    st.subheader("Source Documents")
    for doc in response['source_documents']:
        st.write(doc.metadata.get("source", "Unknown"))

# Optional: show queries generated
if st.checkbox("Show Queries Generated"):
    st.write(response.get("all_queries", []))
