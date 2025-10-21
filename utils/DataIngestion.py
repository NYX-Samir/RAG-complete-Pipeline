
from langchain.document_loaders import (PyPDFLoader,TextLoader,Docx2txtLoader,DirectoryLoader,UnstructuredHTMLLoader,CSVLoader)
from typing import List,Dict,Tuple
import re

class DataIngestion:
    
    @staticmethod
    def load_pdfs(file_path:str):
        loader=PyPDFLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_text(file_path:str):
        loader=TextLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_directory(directory_path:str,glob_pattern:str='**/*.pdf'):
        loader=DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        return loader.load()
        
    @staticmethod
    def load_docx(file_path:str):
        loader=Docx2txtLoader(file_path)
        return loader.load()
    
    @staticmethod
    def preprocess_text(text:str)->str:
        text=re.sub(r"\s+",' ',text)
        text=re.sub(r'[^\w\s\.\?\!\-\:\;]','',text)
        
        return text.strip()