
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, DirectoryLoader,
    UnstructuredHTMLLoader, CSVLoader
)

from langchain.schema import Document
from typing import List
from pathlib import Path
import re

class DataIngestion:
    
    @staticmethod
    def load_pdf(file_path:str):
        return PyPDFLoader(file_path).load()
    
    @staticmethod
    def load_text(file_path:str):
        return TextLoader(file_path).load()
    
    @staticmethod
    def load_html(file_path:str):
        return UnstructuredHTMLLoader(file_path).load()
    
    @staticmethod
    def load_docx(file_path:str):
        return Docx2txtLoader(file_path).load()
    
    @staticmethod
    def load_csv(file_path:str):
        return CSVLoader(file_path).load()
    
    
    @staticmethod
    def load_directory(
        directory_path:str,
        glob_pattern:str="**/*" 
    ) -> List[Document]:
        loader=DirectoryLoader(
            directory_path,
            glob_pattern,
            show_progress=True
        )
        return loader.load()
    
    @staticmethod
    def preprocess_text(text:str)->str:
        if not text:
            return ""
        
        text=re.sub(r"\s"," ",text)
        text=re.sub(r"[^\w\s\.\?\!\-\:\;]","",text)
        return text.strip()
    
    
    @staticmethod
    def normalize_metadata(doc:Document)->Document:
        source=(
            doc.metadata.get("source") or
            doc.metadata.get("file_path") or
            doc.metadata.get("path") or
            "unknown"
        )
        
        doc.metadata = {
            "source":str(source),
            "filename":Path(source).name if source != "unknown" or "Unknown",
            "page":doc.metadata.get("page",None),
        }

        return doc
    
    
    @classmethod
    def ingest(cls,paths:List[str])->List[Document]:
        
        all_docs:List[Document]=[]
        
        for path in paths:
            path_obj=Path(path)
            
            try:
                if path_obj.is_dir():
                    docs=cls.load_directory(str(path_obj))
                    
                    
                else:
                    
                    suffix=path_obj.suffix.lower()
                    
                    if suffix ==".pdf":
                        docs=cls.load_pdf(str(path_obj))
                    elif suffix in [".txt"]:
                        docs=cls.load_text(str(path_obj))
                    elif suffix in [".html",".htm"]:
                        docs=cls.load_html(str(path_obj))
                    elif suffix in [".csv"]:
                        docs=cls.load_csv(str(path_obj))
                    elif suffix in [".docx"]:
                        docs=cls.load_docx(str(path_obj))
                    else:
                        print(f"Skipping unsupported file: {path}")
                        continue
                   
                for doc in docs:
                    doc.page_content =cls.preprocess_text(doc.page_content)
                    doc=cls.normalize_metadata(doc)
                    all_docs.append(doc)
                    
            except Exception as e:
                print(f"Failed to load path: {e}")
                
                
        return all_docs