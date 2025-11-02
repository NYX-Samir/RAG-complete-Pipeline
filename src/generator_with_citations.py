from typing import List
from langchain_community.llms import Ollama

class RAGGenerator:
    
    def __init__(self,model="llama3.2"):
        self.llm=Ollama(model=model,temperature=0)
        
    def generate_with_citations(self,query:str,context_docs:List):
        
        context_text=""
        for i,doc in enumerate(context_docs):
            source=doc.metadata.get('source','Unknown')
            context_text +=f"\n [Source {i+1}]: {source}\n{doc.page_content}\n"
            
        prompt=f"""Answer the question based Only on the context provided.
        Include citations like [Source X] after each claim.
        If the context doesn't contain the answer ,say "I cannot find this information in the provided source."
                                                
        Context:{context_text}
        Question:{query}
        Answer with citations: 
        """
        answer=self.llm.invoke(prompt)
        return answer 
        