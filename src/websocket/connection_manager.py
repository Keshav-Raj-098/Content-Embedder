from collections import defaultdict
from fastapi import WebSocket

class manager:
    def __init__(self):
        self.jobs = defaultdict(set)

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        self.jobs[job_id].add(websocket)

    def disconnect(self, job_id: str, websocket: WebSocket):
        self.jobs[job_id].discard(websocket)
        if not self.jobs[job_id]:
            del self.jobs[job_id]

    async def send(self, job_id: str, message: dict):
        for ws in self.jobs.get(job_id, []):
            await ws.send_json(message)
