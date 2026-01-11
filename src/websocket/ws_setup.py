# ws_setup.py
from .ws_router import WSRouter
from .handlers.cancel import handle_cancel
from .handlers.pause import handle_pause

ws_router = WSRouter()

ws_router.register("cancel", handle_cancel)
ws_router.register("pause", handle_pause)
