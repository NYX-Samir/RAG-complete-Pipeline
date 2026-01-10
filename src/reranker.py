from sentence_transformers import CrossEncoder
from langchain.schema import Document
from typing import List,Tuple,Optional
import torch


class ReRanker:
    
    def __init__(
        self,
        model_name:str="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device:Optional[str]=None,
        batch_size:int=16,
    ): 
        if device is None:
            device= "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Loading re-ranker model on {device}: {model_name}")
        self.model=CrossEncoder(model_name,device=device)
        self.batch_size=batch_size
        print("Re-Ranker ready")
        
    
    def rerank(
        self,
        query:str,
        documents:List[Document],
        top_n:int=5,
    ) -> List[Tuple[Document,float]]:
        
        if not query or not query.strip():
            return []
        
        if not documents:
            return []
        
        pairs=[]
        valid_docs=[]
        
        for doc in documents:
            text=doc.page_content.strip()
            if not text:
                continue
            
            pairs.append([query,text])
            valid_docs.append(doc)
            
        if not pairs:
            return []
        
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        scored_docs= list(zip(valid_docs,scores))
        
        scored_docs.sort(
            key=lambda x :(float (x[1]),x[0].page_content),
            reverse=True,
        )
        
        return scored_docs[:top_n]
            