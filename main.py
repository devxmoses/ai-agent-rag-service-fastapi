import os
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

app = FastAPI(title="FastAPI AI Agent & RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*",]
)

client = AsyncOpenAI()

system_vector_db = []

# Mock database
def get_order_tracking_details(order_id: str):
    mock_db = {
        "98721":{"status": "Delayed by weather", "carrier":"FedEx", "price": 120},
        "11223":{"status": "Shipped", "carrier":"UPS", "price": 45}
    }
    return mock_db.get(order_id, {"error":"Order ID not found"})

# Pydantic schema for incoming client validation
class ChatRequest(BaseModel):
    prompt: str = Field(..., description = "The natural language user question")

#Auto Vectorization upload end point
@app.post("/api/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    global system_vector_db

    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400,detail="Only standard plain text (.txt) files are supported.")

    contents = await file.read()
    raw_text = contents.decode("utf-8")

    # Dynamic text chunking by paragraph layout bounds
    paragraphs = [p.strip() for p in raw_text.split('\n') if len(p.strip()) > 10]

    system_vector_db = []

    try:
        # Generate high-density mathematical vector embeddings asynchronously
        for chunk in paragraphs:
            response = await client.embeddings.create(
                model = "text-embedding-3-small",
                input=chunk
            )
            embedding = response.data[0].embedding
            system_vector_db.append({"text":chunk, "vector":embedding})

        return {"success": True, "chunks": len(paragraphs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding compilation failed: {e}")