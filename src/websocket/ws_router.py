# ws_router.py
class WSRouter:
    def __init__(self):
        self.handlers = {}

    def register(self, event_type: str, handler):
        self.handlers[event_type] = handler

    async def handle(self, job_id, websocket, event):
        event_type = event.get("type")
        if not event_type:
            return

        handler = self.handlers.get(event_type)
        if handler:
            await handler(job_id, websocket, event)
