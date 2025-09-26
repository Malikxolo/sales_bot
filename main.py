from api.chat import router as chat_router
from fastapi import FastAPI
import logging
import os
from starlette.middleware.cors import CORSMiddleware


app = FastAPI(title="🧠❤️ Brain-Heart Agent API", version="1.0.0")
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
)

logging.basicConfig(level=logging.INFO)
@app.get("/")
def read_root():
    return {"message": "Welcome to the 🧠❤️ Brain-Heart Agent API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
    
