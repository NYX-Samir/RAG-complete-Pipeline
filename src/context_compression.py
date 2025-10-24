from langchain.schema import Document
from typing import List

class ContextCompression:
    
    def __init__(self,llm):
        self.llm=llm
        
    def compress_documents(self,query:str,documents:List, max_docs=5):
        
        compressed=[]
        
        for doc in documents[:max_docs]:
            prompt = f"""Extract only the sentences that directly answer or are relevant to the question below. 
                    If no sentence is relevant, output "None". 
                    Do not include unrelated information.

                    Example:
                    Question: What is AI?
                    Document: Artificial intelligence (AI) enables machines to learn. Cars have engines.
                    Relevant sentences: Artificial intelligence (AI) enables machines to learn.

                    Now do the same.

                    Question: {query}

                    Document: {doc.page_content[:1500]}

                    Relevant sentences:"""
            try:
                relevant_text=self.llm.invoke(prompt)
                if relevant_text.strip():
                    compressed.append(Document(
                        page_content=relevant_text.strip(),
                        metadata=doc.metadata
                    ))
            except:
                compressed.append(doc)
        return compressed if compressed else documents[:max_docs]
    