from langchain_community.llms import Ollama

class LocalQueryTransformer:
    
    
    def __init__(self, model="llama3.2"):

        self.llm = Ollama(model=model, temperature=0)
        print(f"Using Ollama model: {model}")
    
    def multi_query(self, original_query: str, num_queries=3):
        
        prompt = f"""Generate {num_queries} different versions of this question to retrieve relevant documents.
                Only output the questions, one per line, without numbering.

                Original question: {original_query}

                Alternative questions:"""
        
        response = self.llm.invoke(prompt)
        
     
        queries = [q.strip() for q in response.split('\n') if q.strip() and '?' in q]
        queries = [original_query] + queries[:num_queries]
        
        return queries
    
    def hyde(self, query: str):
        
        prompt = f"""Write a detailed, factual answer to this question:

                Question: {query}

                Answer:"""
        
        hypothetical_answer = self.llm.invoke(prompt)
        return hypothetical_answer
    
    def step_back(self, query: str):
        
        prompt = f"""Given this specific question, generate a broader, more general question:

                Specific question: {query}

                Broader question:"""
        
        response = self.llm.invoke(prompt)
        return response.strip()