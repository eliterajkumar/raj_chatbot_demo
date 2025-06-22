from fastapi import APIRouter, UploadFile, File, HTTPException, status, Body
from pydantic import BaseModel
from typing import Dict

from ..services import pdf_processor, vector_store, llm_handler

router = APIRouter()

# This will act as our simple, in-memory database for this router
# In a real application, you'd use a persistent database.
in_memory_vector_store: Dict = {}

class QueryRequest(BaseModel):
    question: str

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):
    global in_memory_vector_store
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
    
    try:
        contents = await file.read()
        text = pdf_processor.extract_text_from_pdf(contents)
        chunks = pdf_processor.chunk_text(text, chunk_size=500)
        in_memory_vector_store = vector_store.create_vector_store(chunks)
        return {"message": f"Successfully processed '{file.filename}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

@router.post("/query")
async def query_rag(request: QueryRequest):
    global in_memory_vector_store
    if not in_memory_vector_store:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF first.")
        
    try:
        context_chunks = vector_store.find_similar_chunks(in_memory_vector_store, request.question, top_k=3)
        context = "\n".join(context_chunks)
        
        system_prompt = "You are a helpful assistant. Use the provided context to answer the user's question accurately. If the answer is not in the context, say so."
        answer = llm_handler.get_llm_response(system_prompt, context, request.question)
        
        return {"answer": answer, "source_chunks": context_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get a response: {e}") 