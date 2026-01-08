
import os
from typing import List
from src.vector_embedding  import Embeddings
from src.query_transfromer import LocalQueryTransformer
from src.generator_with_citations import RAGGenerator
from src.chunking import Chunking
from src.documents_ingestion import DataIngestion
from src.context_compression import ContextCompressor
from src.reranker import ReRanker
from src.retrieval import HybridRetriever
from src.evaluation_metrics import RAGEvaluator


class RAGPipeline:
    
    def __init__(self,embedding_model='all-MiniLM-L6-v2',llm_model="llama3.2",use_hybrid=True,use_reranking=True,use_evaluation=True):
        
        print("RAG Pipeline")
        self.embeddings=Embeddings(embedding_model)
        self.query_transformer=LocalQueryTransformer(llm_model)
        self.generator=RAGGenerator(llm_model)
        self.use_hybrid=use_hybrid
        self.use_reranking=use_reranking
        self.use_evaluation=use_evaluation
        
        if use_reranking:
            self.reranker=ReRanker()
        if use_hybrid:
            self.evaluator=RAGEvaluator()
            
        print("Pipeline initialized!\n")
        
    def build_index(self,documents_path:str,chunk_strategy="recursive",save_path=None):
        "Build the index from documents"
        print("="*80)
        print("Building Index")
        print("="*80)
        
        print(f"Loading documents from {documents_path}")
        ingestion=DataIngestion()
        
        if os.path.isfile(documents_path):
            if documents_path.endswith(".pdf"):
                documents=ingestion.load_pdfs(documents_path)
            elif documents_path.endswith(".txt"):
                documents=ingestion.load_text(documents_path)
        else:
            documents=ingestion.load_directory(documents_path)
        print("All Documents Successfully Loaded")
        
        print(f"Chunking with strategy: {chunk_strategy}")
        chunking=Chunking()
        
        if chunk_strategy =="recursive":
            chunks=chunking.recursive_chunking(documents)
        elif chunk_strategy=="sematic":
            chunks=chunking.semantic_chunking(documents,self.embeddings) 
        else:
            chunks=chunking.recursive_chunking(documents)
            
        print(f"Created {len(chunks)} chunks \n")
        
        print("Building Chroma vector index...")
        self.vectorstore = self.embeddings.create_chroma_db(chunks, persist_directory="../chroma_db")
        self.chunks = chunks
        print("Vector index built and persisted\n")

        if self.use_hybrid:
            print("Building hybrid retriever (Vector + BM25)...")
            self.retriever = HybridRetriever(self.vectorstore, chunks)

        print("Index build complete\n")
        
        print("="*80)
        print("Index Build Complete")
        print("="*80)
        
        return self.vectorstore
    
    def query(self,
              question:str,
              use_multi_query=True,
              use_compression=False,
              top_k=10,
              rerank_top_n=5,
              verbose=True):
        if verbose:
            print("\n"+"="*80)
            print(f"Query : {question}")
            print("="*80)
            
        queries=[question]
        if use_multi_query:
            if verbose:
                print("\n Generating query variations...")
            queries=self.query_transformer.multi_query(question,num_queries=2)
            if verbose:
                for i,q in enumerate(queries):
                    print(f"  {i+1}.{q}")
            
        if verbose:
            print("\n Retrieving documents...")
        
        all_docs=[]
        if self.use_hybrid:
            for q in queries:
                docs=self.retriever.retrieve(q,k=top_k)
                all_docs.extend([doc for doc,_ in docs])
        
        else:
            for q in queries:
                docs = self.vectorstore.similarity_search(q,k=top_k)
                all_docs.extend(docs)
                
        seen=set()
        unique_docs=[]
        for doc in all_docs:
            doc_hash=hash(doc.page_content[:100])
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)
                
        if verbose:
            print(f"Retrieved {len(unique_docs)} unique documents")
            
        if self.use_reranking and len(unique_docs)>rerank_top_n:
            if verbose:
                print(f"\n Re-ranking to top {rerank_top_n}....")
            final_docs=self.reranker.rerank(question,unique_docs[:20],top_n=rerank_top_n)
            if verbose:
                print(f"Re-ranked documents")
        else:
            final_docs=unique_docs[:rerank_top_n]
                
        if use_compression:
            if verbose:
                print(f"\n Compressing context...")
                
            compressor=ContextCompressor(self.query_transformer.llm)
            final_docs=compressor.compress_documents(question,final_docs)
            
        if verbose:
            print(f"\n Generating Answer \n")
            
        answer=self.generator.generate_with_citations(question,final_docs)
        
        if verbose:
            print("="*80)
            print("ANSWER:")
            print("="*80)
            print(answer)
            print("\n" + "="*80)
            print(f" Used {len(final_docs)} source documents")
            print("="*80)
            print()
            
        return {
            'answer': answer,
            'source_documents': final_docs,
            'num_sources': len(final_docs),
            'all_queries': queries
            }
        
        