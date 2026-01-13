from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from typing import List,Tuple,Dict
import numpy as np 

class HybridRetriever:
    
    def __init__(self,vectorstore,documents:List[Document]):
        
        self.vectorstore=vectorstore
        self.documents=documents
        self._build_bm25(documents)
        
        print(f"Hybrid Retriever initialized with {len(documents)} documents")
        
    
    @staticmethod
    def _tokenize(text:str)->List[str]:
        return text.lower().split()
    
    @staticmethod
    def _doc_uid(doc:Document)->str:
        
        src=doc.metadata.get("source","unknown")
        page=doc.metadata.get("page","na")
        return f"{src}::page={page}::hash={hash(doc.page_content)}"
    
    
    
    @staticmethod
    def _normalize(scores:np.ndarray)->np.ndarray:
        if len(scores)==0:
            return scores
        
        min_s,max_s=scores.min(),scores.max()
        if max_s>min_s:
            return (scores-min_s)/(max_s-min_s)
        
        return scores
    
    
    def _build_bm25(self,documents:List[Document])->None:
        tokenized_docs=[
            self._tokenize(doc.page_content)
            for doc in documents
        ]
        self.bm25=BM25Okapi(tokenized_docs)
        
        
    def refresh_documents(self,documents:List[Document])->None:
        
        self.documents=documents
        self._build_bm25(documents)
        
        
    def retrieve(
        self,
        query:str,
        k:int=10,
        alpha:float=0.5,
        bm25_k:int=50,
    ) -> List[Tuple[Document, float]]:
        
        if not query or not query.strip():
            return []
        
        # Dense retrieval
        dense_results=self.vectorstore.similarity_search_with_score(
            query,
            k=k*2
        )
        
        dense_docs= [doc for doc,_ in dense_results]
        dense_distances=np.array([score for _,score in dense_results])
        
        dense_scores=1/(1 + dense_distances)
        dense_scores=self._normalize(dense_scores)
        
        # BM25 retrieval (top-k only)
        
        tokenized_query= self._tokenize(query)
        bm25_raw_score=self.bm25.get_scores(tokenized_query)
        
        top_bm25_idx=np.argsort(bm25_raw_score)[::-1][:bm25_k]
        bm25_scores=bm25_raw_score[top_bm25_idx]
        bm25_scores=self._normalize(bm25_scores)
        
        score_map: Dict[str, Dict] = {}
        
        for i,doc in enumerate(dense_docs):
            uid=self._doc_uid(doc)
            score_map[uid]={
                "doc":doc,
                "score":alpha*dense_scores[i]
            }
            
        for rank,idx in enumerate(top_bm25_idx):
            doc =self.documents[idx]
            uid=self._doc_uid(doc)
            
            sparse_score=(1-alpha)*bm25_scores[rank]
            
            if uid in score_map:
                score_map[uid]["score"] += sparse_score
            else:
                score_map[uid]={
                    "doc":doc,
                    "score":sparse_score,
                }
                
                
        ranked= sorted(
            score_map.values(),
            key=lambda x : (x["score"],x["doc"].page_content),
            reverse=True,
        )[:k]
        
        return [(item["doc"],float(item["score"])) for item in ranked]
    
    
    