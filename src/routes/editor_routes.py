from fastapi import APIRouter, Request,Depends
from src.controllers import editor_controller



router = APIRouter(prefix="/api/editor", tags=["users"])



@router.get("/test")
async def get_user(request: Request):
    return {"status": "ok", "message": "Editor routes are working 🚀"}
  
  

@router.post("/set-name/{name}")
async def set_name(name: str):
    return await editor_controller.add(name)


@router.get("/get-name")
async def get_name():
    return await editor_controller.get()