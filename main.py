from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import PROJECT_NAME, VERSION, DEBUG
from src.api.endpoints import router as api_router

app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    debug=DEBUG
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "Light-RAG Knowledge Graph API",
        "version": VERSION
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)