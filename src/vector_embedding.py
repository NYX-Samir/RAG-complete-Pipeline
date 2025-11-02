from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

class Embeddings:
    
    def __init__(self,model_name='all-MiniLM-L6-v2'):
        
        self.embeddings=HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{model_name}",
            model_kwargs={'device':'cuda'},
            encode_kwargs={'normalize_embeddings':True}
        )
        
    def create_chroma_db(self,chunks,persist_directory="../chroma_db"):
        vectordb=Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        return vectordb

