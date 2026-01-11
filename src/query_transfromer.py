from langchain_community.llms import  HuggingFaceHub

class QueryTransformer:
    
    def __init__(
        self,
        repo_id:str="meta-llama/Llama-2-7b-chat-hf",
        temperature:float=0.0,
        max_new_tokens:int=256,
    ):
        print(f"Using HuggingFace API model {repo_id}")
        
        
        self.llm =HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={
                "temperature":temperature,
                "max_new_tokens":max_new_tokens,
                "do_sample":False,
            }
        )
        print("HuggingFace API LLM ready")
        
    
    def multi_query(
        self,
        original_query:str,
        num_queries:int=3
    ):
        if not original_query or not original_query.strip():
            return []
        
        
        prompt=f"""
Generate {num_queries} different versions of this question to retrieve relavant documents.
Only output the question ,one per line ,without numbering.

Original_query Question:
{original_query}

Alternate Questions:
""".strip()
        
        response =self.llm.invoke(prompt)
        text =str(response).strip()
        
        queries= [
            q.strip()
            for q in text.split("\n")
            if q.strip() and len(q.strip()) >5
        ]
        
        return [original_query]+queries[:num_queries]
    
    
    def hyde(
        self,
        query:str
    )->str:
        
        if not query or not query.strip():
            return ""
        
        
        prompt=f"""
Write a concise ,factual answer to this question.

Question :
{query}

Answer:
""".strip()
        
        
        response=self.llm.invoke(prompt)
        return str(response).strip()
    
    
    
    def step_back(
        self,
        query:str
    )->str:
        
        if not query or not query.strip():
            return ""
        
        
        prompt=f"""
Given this specific question ,generate a broader and more general question.

Specific question:
{query}

Broader question:
""".strip()
        
        
        response=self.llm.invoke(prompt)
        return str(response).strip()
    
    
    