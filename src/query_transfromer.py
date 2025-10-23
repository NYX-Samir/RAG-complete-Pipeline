from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

class QueryTransformer:
    
    def __init__(self,model_name="meta-llama/Llama-3.1-8B"):
        
        load_dotenv()
        hf_token=os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError("HF_TOKEN is not found!!!!")
        
        self.client=HuggingFaceEndpoint(
            repo_id=model_name,
            task="text-generation",
            huggingfacehub_api_token=hf_token
        )
        self.model_name=model_name
        print(f"Using HF model : {model_name}")
        
    def _generate(self,prompt:str,max_new_token=256):
        
        reponse=self.client.invoke(prompt,max_new_tokens=max_new_token)
        return reponse.strip()
    
    def multi_query(self,original_query:str,num_queries=3):
        prompt=f"""Generate {num_queries} different versions of this question to retrieve relevant documents.
        Only output the questions, one per line,without numbering.
        Original question :{original_query}
        Alternative question :"""
        
        response = self._generate(prompt)
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        queries=[original_query]+queries[:num_queries]
        return queries
    
    def hyde(self,query:str):
        prompt=f"""Write a detailed,factual answer to this question:
        Question : {query}
        Answer :"""
        return self._generate(prompt)
    
    
    def step_back(self,query:str):
        prompt=f"""Give this specific question , generate a broader,more general question:
        Specific question :{query}
        Broader question :"""
        
        return self._generate(prompt)