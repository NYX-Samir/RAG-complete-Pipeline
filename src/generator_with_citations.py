from typing import List
from langchain.schema import Document
from langchain_community.llms import HuggingFaceHub


class RAGGenerator:
    
    def __init__(
        self,
        repo_id:str="meta-llama/Llama-2-7b-chat-hf",
        temperature:float=0.0,
        max_new_tokens:int=512,
        max_content_chars:int=6000,
    ):
        print(f"Using HuggingFace API generator model :{repo_id}")
        
        self.llm=HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={
                "temperature":temperature,
                "max_new_tokens":max_new_tokens,
                "do_sample":False,
            },
        )
        
        self.max_content_chars=max_content_chars
        print("RAG Generator Ready")
        
        
    @staticmethod
    def _extract_text(response)->str:
        
        if response is None:
            return ""
        
        if hasattr(response,"content"):
            return str(response.content).strip()
        
        return str(response).strip()
    
    
    def generate_with_citations(
        self,
        query:str,
        context_docs:List[Document],
    )->str:
        
        if not query or not query.strip():
            return "Invalid query. "
        
        if not context_docs:
            return "I cannot find this information in the provided source."
        
        
        context_text=""
        total_chars=0
        
        for i, doc in enumerate(context_docs):
            source=doc.metadata.get("source","Unknown")
            block=f"\n[Source {i+1}]: {source}\n{doc.page_content}\n"
            
            total_chars += len(block)
            if total_chars > self.max_content_chars:
                break
            
            
            context_text +=block
            
        
        prompt=f"""
Answer the question using Only the information in the context below.
Cite each factual statement using the format [Source X].
If the answer is not present in the context ,say exactly:
"I cannot find this information in the provided source."

Context:
{context_text}

Question:
{query}

Answer with citations:
""".strip()
        
        
        try:
            response=self.llm.invoke(prompt)
            answer=self._extract_text(response)
            
            
            if not answer:
                return "I cannot find this information in the provided source."
            
            return answer
        
        except Exception as e:
            return f"Generation failed :{str(e)}"

            