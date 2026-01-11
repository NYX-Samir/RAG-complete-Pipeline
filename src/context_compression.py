from langchain.schema import Document
from typing import List
import copy

class ContextCompression:
    
    def __init__(self,llm,max_chars:int=1500):
        
        self.llm=llm
        self.max_chars=max_chars
        
        
        
    @staticmethod
    def _extract_text(response)->str:
        
        if response is None:
            return ""
        
        if hasattr(response,"content"):
            return str(response.content).strip()
        
        return str(response).strip()
    
    
    def compress_documents(
        self,
        query:str,
        documents:List[Document],
        max_docs:int=5,
    )->List[Document]:
        
        if not query or not query.strip():
            return []
        
        if not documents:
            return []
        
        compressed:List[Document]=[]
        
        for doc in documents[:max_docs]:
            text=doc.page_content.strip()
            if not text:
                continue
            
                
            prompt=f"""
Extract only the sentences that directly answer or are relevant to the question below.
If no sentence is relavant, output exactly:None.
Do not add explainations or extra text.
            
Question:
{query}
            
Document:
{text[:self.max_chars]}
Relevant sentences:
""".strip()
            
            
            try:
                response=self.llm.invoke(prompt)
                relevant_text=self._extract_text(response)
                
                if not relevant_text:
                    continue
                
                if relevant_text.lower() == "none":
                    continue
                
                compressed.append(
                    Document(
                        page_content=relevant_text,
                        metadata=copy.deepcopy(doc.metadata),
                    )
                )
                
                
            except Exception as e:
                
                compressed.append(
                    Document(
                        page_content=text,
                        metadata=copy.deepcopy(doc.metadata),
                    )
                )
                
                
        if not compressed:
            return documents[:max_docs]
        
        return compressed
    
    
                