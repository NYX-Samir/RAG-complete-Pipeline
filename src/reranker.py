from sentence_transformers import CrossEncoder

class ReRanker:
    
    def __init__(self,model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        
        print(f"Loading re-ranker model: {model_name}...")
        self.model=CrossEncoder(model_name)
        print("Re-Ranker loaded")
        
    def rerank(self,query:str,documents:List,top_n=5):
        
        pairs=[[query,doc.page_content] for doc in documents]
        
        scores=self.model.predict(pairs)
        
        scored_docs=list(zip(documents,scores))
        scored_docs.sort(key=lambda x : x[1],reverse=True)
        
        return [doc for doc,_ in scored_docs[:top_n]]
    
    