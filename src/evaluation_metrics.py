from typing import List,Union,Set,Tuple
from langchain_core.documents import Document

class RAGEvaluator:
    
    
    @staticmethod
    def _unwrap_docs(
        retrieved_docs: List[Union[Document, Tuple[Document, float]]]
    )->List[Document]:
        
        docs=[]
        for item in retrieved_docs:
            if isinstance(item,tuple):
                docs.append(item[0])
            else:
                docs.append(item)
                
        return docs
    
    
    
    @staticmethod
    def _doc_uid(doc:Document)->str:
       
       src =doc.metadata.get("source","Unknown")
       page=doc.metadata.get("page","na")
       
       return f"{src}::page={page}::hash={hash(doc.page_content)}"
   
   
   
    @classmethod
    def calculate_mrr(
        cls,
        retrieved_docs:List[Union[Document,Tuple[Document,float]]],
        relevant_doc_uids:Set[str],
    )->float:
        
        docs=cls._unwrap_docs(retrieved_docs)
        
        for i ,doc in enumerate(docs):
            uid=cls._doc_uid(doc)
            if uid in relevant_doc_uids:
                return 1.0/(i+1)
            
        return 0.0
        
        
    @classmethod
    def calculate_recall_at_k(
        cls,
        retrieved_docs:List[Union[Document,Tuple[Document,float]]],
        relevant_doc_uids:Set[str],
        k:int=5,
    )->float:
        
        if not relevant_doc_uids:
            return 0.0
        
        docs=cls._unwrap_docs(retrieved_docs)[:k]
        retrieved_uids={cls._doc_uid(doc) for doc in docs}
        
        relevant_retrieved=retrieved_uids & relevant_doc_uids
        return len(relevant_retrieved)/len(relevant_doc_uids)
    
    
    @classmethod
    def calculate_precision_at_k(
        cls,
        retrieved_docs: List[Union[Document, Tuple[Document, float]]],
        relevant_doc_uids: Set[str],
        k: int = 5,
    ) -> float:

        docs = cls._unwrap_docs(retrieved_docs)[:k]

        if not docs:
            return 0.0

        retrieved_uids = {cls._doc_uid(doc) for doc in docs}
        relevant_retrieved = retrieved_uids & relevant_doc_uids

        denominator = min(k, len(docs))
        return len(relevant_retrieved) / denominator