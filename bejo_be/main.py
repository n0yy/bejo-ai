from fastapi import FastAPI
from app.api.chat import router as chat_router

app = FastAPI(title="BEJO API", description="BEJO API", version="1.0.0")
app.include_router(chat_router)
@app.get("/")
async def root():
    return {"message": "Welcome to the BEJO API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)