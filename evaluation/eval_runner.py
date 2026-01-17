import time
from statistics import mean

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import RAGPipeline
from src.evaluation_metrics import RAGEvaluator
from evaluation.eval_dataset import EVALUATION_DATASET

TOP_K=5
INSPECTION_MODE=False

def run_evaluation():
    pipeline=RAGPipeline(
        data_paths=["./data"],
        persist_dir="./chroma_db",
        chunking_mode="recursive",
        enable_compression=False,
        enable_rerank=False,
        top_k=TOP_K,
        verbose=False,
    )
    
    pipeline.build_index(rebuild=False)
    pipeline.load_models()
    
    recall_scores =[]
    precision_scores=[]
    mrr_scores=[]
    latencies=[]
    
    
    for item in EVALUATION_DATASET:
        query=item["query"]
        relevant_uids=item["relevant_uids"]
        
        print("\n"+"="*80)
        print("Query: ",query)
        
        start=time.time()
        retrieved = pipeline.retrieve_for_evaluation(query, k=TOP_K)
        latency=time.time() -start
        
        if INSPECTION_MODE:
            for doc,score in retrieved:
                uid= RAGEvaluator._doc_uid(doc)
                print("\nUID:", uid)
                print("CONTENT PREVIEW:")
                print(doc.page_content[:400])
                print("-" * 80)
                
                
        recall=RAGEvaluator.calculate_recall_at_k(
            retrieved_docs=retrieved,
            relevant_doc_uids=relevant_uids,
            k=TOP_K
        )
        
        precision=RAGEvaluator.calculate_precision_at_k(
            retrieved_docs=retrieved,
            relevant_doc_uids=relevant_uids,
            k=TOP_K
        )
        
        mrr=RAGEvaluator.calculate_mrr(
            retrieved_docs=retrieved,
            relevant_doc_uids=relevant_uids
        )
        
        
        recall_scores.append(recall)
        precision_scores.append(precision)
        mrr_scores.append(mrr)
        
        latencies.append(latency)
                
        print(f"Recall@{TOP_K}: {recall:.3f}")
        print(f"Precision@{TOP_K}: {precision:.3f}")
        print(f"MRR: {mrr:.3f}")
        print(f"Latency: {latency:.2f}s")
        
    print("\n===== AGGREGATED RESULTS =====")
    print(f"Mean Recall@{TOP_K}: {mean(recall_scores):.3f}")
    print(f"Mean Precision@{TOP_K}: {mean(precision_scores):.3f}")
    print(f"Mean MRR: {mean(mrr_scores):.3f}")
    print(f"Mean Latency: {mean(latencies):.2f}s")        
        
    
if __name__ == "__main__":
    run_evaluation()