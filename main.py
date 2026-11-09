from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from supabase import create_client, Client
from fastembed import TextEmbedding
from huggingface_hub import snapshot_download

app = FastAPI(title="RAG Service for VK Bot")

# Скачиваем модель вручную и загружаем её локально
print("Скачивание и загрузка модели...")
model_dir = snapshot_download(repo_id="cointegrated/rubert-tiny2")
model = TextEmbedding(model_name=model_dir)  # Передаем локальный путь

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
        query_embedding = list(model.embed([request.question]))[0].tolist()
        
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
        
        if not chunks:
            return {"status": "success", "chunks_added": 0}

        embeddings = list(model.embed(chunks))
        
        data_to_insert = []
        for i, chunk in enumerate(chunks):
            data_to_insert.append({
                "document_name": request.document_name,
                "content": chunk,
                "embedding": embeddings[i].tolist()
            })
            
        supabase.table("document_chunks").insert(data_to_insert).execute()
        return {"status": "success", "chunks_added": len(data_to_insert)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
