from api.chat import router as chat_router, lifespan
from fastapi import FastAPI
import logging
import os
from starlette.middleware.cors import CORSMiddleware
from core.logging_config import setup_logging

# Initialize logging system with file rotation
setup_logging(log_dir="logs", log_file="api.log", max_bytes=10_000_000, backup_count=5)

app = FastAPI(title="Sales Agent API", version="1.0.0", lifespan=lifespan)
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Agent API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
    
