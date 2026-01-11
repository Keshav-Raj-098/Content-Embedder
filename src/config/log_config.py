import logging
from rich.logging import RichHandler
from datetime import datetime

# Silence noisy libraries BEFORE logger creation
logging.getLogger("pymongo").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("langsmith.client").setLevel(logging.CRITICAL)
logging.getLogger("python_multipart.multipart").setLevel(logging.CRITICAL)
logging.getLogger("grpc._cython.cygrpc").setLevel(logging.CRITICAL)


class PrettyFormatter(logging.Formatter):
    def format(self, record):
        msg_lines = record.getMessage().splitlines()

        title = f"[{record.name}:{record.levelname}]  {msg_lines[0]}"
        body = "\n".join(msg_lines[1:]) if len(msg_lines) > 1 else ""

        dt = datetime.fromtimestamp(record.created)
        now = dt.strftime("%d-%m-%Y %H:%M:%S")

        top_bar = "─" * 40 + f" {now} " + "─" * 40
        divider = "------------------------------------------------------------"

        if body:
            return f"{top_bar}\n{title}\n{divider}\n{body}"
        return f"{top_bar}\n{title}"




import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

_rich_handler = None  # singleton handler


def get_logger(name="MyApp"):
    global _rich_handler

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        if _rich_handler is None:
            _rich_handler = RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_time=False,
                show_level=False,
                show_path=False,
            )
            _rich_handler.setFormatter(PrettyFormatter())

        logger.addHandler(_rich_handler)

    logger.propagate = False
    return logger
