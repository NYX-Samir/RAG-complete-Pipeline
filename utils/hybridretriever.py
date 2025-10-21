from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    
    def __init__(self,vectorstore,documents):
        self.vectorstore=vectorstore
        self.documents=documents
        
        tokenized_docs=[doc.page_content.lower().split() for doc in documents]
        self.bm25=BM25Okapi(tokenized_docs)
        print(f"Hybrid Retriever ready with {len(documents)} documents")
        
        
    def retrieve(self,query:str,k=10,alpha=0.5):
        
        # Vector Search 
        dense_results=self.vectorstore.similarity_search_with_score(query,k=k*2)
        
        # BM25 Search
        tokenized_query=query.lower().split()
        bm25_scores=self.bm25.get_scores(tokenized_query)
        
        #Normalized Scores between 0-1 
        dense_scores=np.array([1/(1+score) for _,score in dense_results])
        if dense_scores.max()>dense_scores.min():
            dense_scores=(dense_scores-dense_scores.min())/(dense_scores.max()-dense_scores.min())
            
        if bm25_scores.max()>bm25_scores.min():
            bm25_scores=(bm25_scores-bm25_scores.min())/(bm25_scores.max()-bm25_scores.min())
            
        doc_scores={}
        # ADD dense scores
        for i, (doc, _) in enumerate(dense_results):
            doc_id=id(doc)
            doc_scores[doc_id]={'doc':doc,'score':alpha*dense_scores[i]}
            
        #ADD Sparse scores
        for i,doc in enumerate(self.documents):
            doc_id=id(doc)
            if doc_id in doc_scores:
                doc_scores[doc_id]['score']+=(1-alpha)*bm25_scores[i]
            else:
                doc_scores[doc_id]={'doc':doc,'score':(1-alpha)*bm25_scores[i]}
                
        # Sort by combined score 
        sorted_docs=sorted(doc_scores.values(),key=lambda x:x['score'],reverse=True)[:k]
        
        return[(item['doc'],item['score']) for item in sorted_docs]