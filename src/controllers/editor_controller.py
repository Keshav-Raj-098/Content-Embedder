from fastapi.responses import JSONResponse
from fastapi import status

from src.state import app_state
from src.config.log_config import get_logger

logger = get_logger("EDITOR")


async def add(name: str) -> str:
    try:
        app_state.name = name
        
        logger.error(f"Name set to: {name}")
        
        return JSONResponse(
                content={"message": "Name set successfully"},
                status_code=status.HTTP_201_CREATED
            )
    
    except Exception as e:
        logger.error(f"Error in add function: {e}")
        return JSONResponse(
                content={"message": "Internal Server Error"},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )




async def get() -> str:
    try:
        name = app_state.name 

        return JSONResponse(
                content={"message": f"Current name is {name}"},
                status_code=status.HTTP_200_OK
            )
    
    except Exception as e:
        logger.error(f"Error in get function: {e}")