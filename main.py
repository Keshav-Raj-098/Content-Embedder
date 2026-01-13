from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.websocket.connection_manager import manager
from src.websocket.ws_router import WSRouter as ws_router
from src.websocket import ws_setup  # 👈 side-effect import (MANDATORY)
from fastapi.staticfiles import StaticFiles
from huggingface_hub import login
import os



login(token=os.environ.get("HUGGINGFACE_API_KEY"))


# Routes Imports
from src.routes.editor_routes import router as editor_routers

app = FastAPI(
    title="Content Embedder API",
    description="FastAPI server for Content Embedder application",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(editor_routers)

@app.get("/")
def root():
    return {"status": "ok", "message": "FastAPI server is running 🚀"}

app.mount("/media", StaticFiles(directory="media"), name="media")

@app.websocket("/ws/editor/{job_id}")
async def editor_ws(websocket: WebSocket, job_id: str):
    await manager.connect(job_id, websocket)

    try:
        while True:
            event = await websocket.receive_json()
            await ws_router.handle(job_id, websocket, event)

    except WebSocketDisconnect:
        manager.disconnect(job_id, websocket)
