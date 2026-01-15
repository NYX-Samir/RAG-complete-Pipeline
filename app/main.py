from fastapi import FastAPI,HTTPException
from app.schemas import QueryRequest,QueryResponse
from typing import Optional
from src.pipeline import RAGPipeline

app=FastAPI(
    title="Production RAG API",
    version="1.0.0"
)

# Global pipeline instance
pipeline:Optional[RAGPipeline]=None

@app.on_event("startup")
def startup_event():
    
    global pipeline
    
    try:
        pipeline=RAGPipeline(
            data_paths=["./data"],
            persist_dir="./chroma_db",
            chunking_mode="recursive",
            enable_rerank=True,
            enable_compression=True,
            top_k=5,
            verbose=False     
        )
        
        pipeline.build_index()
        pipeline.load_models()
        
    except Exception as e:
        pipeline=None
        raise RuntimeError(f"Startup failed: {e}")
    
@app.get("/health")
def health_check():
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )
    return {"status":"ok"}


@app.post("/query",response_model=QueryResponse)
def query_rag(request:QueryRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )
        
    try :
        answer=pipeline.run(request.query)
        return QueryResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        
@app.post("/rebuild-index")
def rebuild_index():
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )


    try:
        pipeline.rebuild_index()
        return {"status":"index rebuild"}
    
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        