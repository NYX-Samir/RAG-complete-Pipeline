from src.rag_pipeline import RAGPipeline
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)

if __name__ == "__main__":
    
    rag = RAGPipeline(
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama3.2",
        use_hybrid=True,
        use_reranking=True
    )

    
    documents_path = r"C:\Users\evilk\OneDrive\Desktop\Projects\RAG-Complete-Pipeline\data"  # folder or single file
    index_save_path = "./index_data"

    rag.build_index(
        documents_path=documents_path,
        chunk_strategy="recursive",
        save_path=index_save_path
    )

    question = "What is the procedure for expense reimbursement for employees?"
    response = rag.query(
        question=question,
        use_multi_query=True,
        use_compression=False,
        top_k=5,
        rerank_top_n=3,
        verbose=False
    )

    print("\nFINAL ANSWER:\n")
    print(response["answer"])
    print("\nSOURCE DOCS USED:", response["num_sources"])
    print("\nQUERIES GENERATED:", response["all_queries"])


