from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

app = FastAPI(title="RAG Service for VK Bot")

# Загружаем модель
print("Загрузка модели rubert-tiny2...")
model = SentenceTransformer('cointegrated/rubert-tiny2')

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

class SearchRequest(BaseModel):
    question: str
    top_k: int = 3

class UploadRequest(BaseModel):
    document_name: str
    content: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAG Service is running"}

@app.post("/search")
def search(request: SearchRequest):
    try:
        query_embedding = model.encode(request.question).tolist()
        response = supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.3,
                'match_count': request.top_k
            }
        ).execute()
        chunks = [item['content'] for item in response.data]
        return {"context": "\n\n".join(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
def upload(request: UploadRequest):
    try:
        chunks = [request.content[i:i+500] for i in range(0, len(request.content), 500)]
        chunks = [c for c in chunks if c.strip()]
        
        data_to_insert = []
        for chunk in chunks:
            embedding = model.encode(chunk).tolist()
            data_to_insert.append({
                "document_name": request.document_name,
                "content": chunk,
                "embedding": embedding
            })
            
        supabase.table("document_chunks").insert(data_to_insert).execute()
        return {"status": "success", "chunks_added": len(data_to_insert)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
