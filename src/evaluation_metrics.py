class RAGEvaluator:
    
    @staticmethod
    def calculate_mrr(retrieved_docs,relevant_doc_ids):
        for i,doc in enumerate(retrieved_docs):
            if doc.metadata.get('doc_id') in relevant_doc_ids:
                return 1/(i+1)
        return 0
    
    @staticmethod
    def calculate_recall_at_k(retrieved_docs,relevant_doc_ids,k=5):
        retrieved_ids=[doc.metadata.get('doc_id') for doc in retrieved_docs[:k]]
        relevant_retrieved=set(retrieved_ids)& set(relevant_doc_ids)
        return len(relevant_retrieved)/len(relevant_doc_ids) if relevant_doc_ids else 0
    
    @staticmethod
    def calculate_precision_at_k(retrieved_docs,relevant_doc_ids,k=5):
        retrieved_ids=[doc.metadata.get("doc_id") for doc in retrieved_docs[:k]]
        relevant_retrieved=set(retrieved_ids)&set(relevant_doc_ids)
        return len(relevant_retrieved)/k if k >0 else 0        