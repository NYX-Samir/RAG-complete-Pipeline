from langchain.text_splitter import (RecursiveCharacterTextSplitter,CharacterTextSplitter)
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import numpy as np

class Chunking:
     
    @staticmethod
    def recursive_chunking(documents,chunk_size=1000,chunk_overlap=200):
        textSplitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        return textSplitter.split_documents(documents)
    
    @staticmethod
    def semantic_chunking(documents,embedding,chunk_size=1000):
         chunks=[]
         model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
         
         for doc in documents:
             sentences=re.split(r'(?<=[.!?])\s+',doc.page_content)
             
             if len(sentences)<=1:
                 chunks.append(doc)
                 continue
             
             embedding_array=model.encode(sentences)
             
             similarities=[]
             for i in range(len(embedding_array)-1):
                 sim=np.dot(embedding_array[i],embedding_array[i+1])
                 similarities.append(sim)
                 
             threshold=np.percentile(similarities,30)
             
             current_chunk=[]
             for i,sentence in enumerate(sentences):
                 current_chunk.append(sentence)
                 
                 if i <len(similarities) and similarities[i]<threshold:
                     chunk_text=' '.join(current_chunk)
                     if len(chunk_text)>chunk_size:
                         chunks.append(Document(
                             page_content=chunk_text,
                             metadata=doc.metadata
                         ))
                         current_chunk=[]
                         
             if current_chunk:
                chunks.append(Document(
                    page_content=' '.join(current_chunk),
                    metadata=doc.metadata
                ))
         return chunks