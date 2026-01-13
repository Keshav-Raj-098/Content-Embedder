from fastapi import APIRouter, Request,HTTPException,status,UploadFile, File
from fastapi.responses import JSONResponse,StreamingResponse
import io

from src.controllers import editor_controller
from src.schemas.editor_request import * 
from src.controllers.exceptions import VideoLoadError

import tempfile
import shutil
import uuid
import os


UPLOAD_DIR = "uploads/videos"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
os.makedirs(UPLOAD_DIR, exist_ok=True)




UPLOAD_AD_DIR = "uploads/ads"
ALLOWED_AD_FILE_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv",".png", ".jpg", ".jpeg", ".bmp"}
MAX_AD_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
os.makedirs(UPLOAD_AD_DIR, exist_ok=True)

router = APIRouter(prefix="/api/editor", tags=["users"])



@router.get("/test")
async def get_user(request: Request):
    return {"status": "ok", "message": "Editor routes are working 🚀"}
  
  

@router.post("/set-name/{name}")
async def set_name(name: str):
    return await editor_controller.add(name)



@router.post("/load-video")
async def load_video_endpoint(file: UploadFile = File(...)):
    """
    Load a video into the editor state.
    """
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No video file uploaded",
        )

    # ---- validate filename extension ----
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}",
        )

    # ---- generate safe unique filename ----
    filename = f"{uuid.uuid4().hex}{ext}"
    video_path = os.path.join(UPLOAD_DIR, filename)

    # ---- enforce file size while writing ----
    size = 0
    try:
        with open(video_path, "wb") as buffer:
            while chunk := file.file.read(1024 * 1024):  # 1MB chunks
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail="File too large",
                    )
                buffer.write(chunk)
    finally:
        file.file.close()

        
    # -------------------------------
    try:
        result = editor_controller.load_video(video_path)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result)

    except VideoLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while loading video",
        )
        




@router.post("/load-ad")
async def load_video_endpoint(file: UploadFile = File(...)):
    """
    Load a add file into the editor state.
    """
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Ad file uploaded",
        )

    # ---- validate filename extension ----
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_AD_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}",
        )

    # ---- generate safe unique filename ----
    filename = f"{uuid.uuid4().hex}{ext}"
    ad_file_path = os.path.join(UPLOAD_DIR, filename)

    # ---- enforce file size while writing ----
    size = 0
    try:
        with open(ad_file_path, "wb") as buffer:
            while chunk := file.file.read(1024 * 1024):  # 1MB chunks
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail="File too large",
                    )
                buffer.write(chunk)
    finally:
        file.file.close()

        
    # -------------------------------
    try:
        result = editor_controller.load_ad_file(ad_file_path)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
            "data": result,
        })

    except VideoLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while loading ad file",
        )
        




@router.post("/set-placement")
async def set_placement(req: PlacementRequest):
   try:
       if not req.quad or len(req.quad) != 4:
           raise HTTPException(
               status_code=status.HTTP_400_BAD_REQUEST,
               detail="Invalid quad data",
           )
           
       editor_controller.set_placement(req.quad)
       
       return JSONResponse(
           status_code=status.HTTP_200_OK,
           content=None
           )
    
   except Exception as e:
       print("ERROR IN SET PLACEMENT:", str(e))
       raise HTTPException(
           status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="Unexpected error while setting placement",
       ) 


@router.get("/get-placement")
async def get_placement():
    try:
        quad = editor_controller.get_placement()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
            "data": {
                "quad": quad.tolist() if quad is not None else None,
            },
        })

    except Exception as e:
        print("ERROR IN SET PLACEMENT:", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while fetching placement",
        )

   
@router.get("/input-frame/{idx}")
async def get_input_frame(idx: int):
    try:
        frame_url = editor_controller.get_frame_url(idx)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"frame_url": frame_url},
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frame not found",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while fetching frame",
        )


   
@router.post("/reset")
async def reset_editor():
    """
    Reset the editor state.
    """
    try:
        # editor_controller.reset_app_state()
        pass
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
            "data": "Editor state reset successfully",
        })

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while resetting editor state",
        )
        
        
@router.post("/track-placement")
async def track_placement_endpoint():
    """
    Track the placement quad across all video frames using SAM3.
    
    This uses the quad from state.placement.quad and tracks it through
    all frames in the loaded video.
    
    Returns:
        JSON with tracking results including masks directory path
    """
    try:
        result = editor_controller.track_placement_quad()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "data": {
                    "total_frames": result["total_frames"],
                    "masks_dir": result["masks_dir"],
                    "obj_id": result["obj_id"],
                },
                "message": f"Successfully tracked quad across {result['total_frames']} frames"
            }
        )
    
    except VideoLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during tracking: {str(e)}",
        )


@router.post("/track-custom-box")
async def track_custom_box_endpoint(box: list[float]):
    """
    Track a custom bounding box across video frames.
    
    Args:
        box: Array of 4 floats [x_min, y_min, x_max, y_max]
    
    Returns:
        JSON with tracking results
    """
    try:
        if len(box) != 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Box must have exactly 4 values [x_min, y_min, x_max, y_max]",
            )
        
        result = editor_controller.track_custom_box(box=box)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "data": {
                    "total_frames": result["total_frames"],
                    "masks_dir": result["masks_dir"],
                    "obj_id": result["obj_id"],
                },
                "message": f"Successfully tracked box across {result['total_frames']} frames"
            }
        )
    
    except VideoLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during tracking: {str(e)}",
        )


@router.get("/tracking-mask/{frame_idx}")
async def get_tracking_mask_endpoint(frame_idx: int, masks_dir: str):
    """
    Get a specific tracking mask as JSON.
    
    Args:
        frame_idx: Frame index to retrieve
        masks_dir: Directory containing the masks
    
    Returns:
        JSON with mask shape and statistics
    """
    try:
        mask = editor_controller.load_tracking_mask(frame_idx, masks_dir)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "data": {
                    "frame_idx": frame_idx,
                    "shape": list(mask.shape),
                    "has_mask": bool(mask.any()),
                    "mask_coverage": float(mask.sum() / mask.size),
                }
            }
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    
    except IndexError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error loading mask: {str(e)}",
        )


@router.get("/visualize-tracking/{frame_idx}")
async def visualize_tracking_endpoint(
    frame_idx: int, 
    masks_dir: str,
    color_r: int = 0,
    color_g: int = 255,
    color_b: int = 0,
    alpha: float = 0.5
):
    """
    Visualize tracking result on a specific frame.
    
    Args:
        frame_idx: Frame index to visualize
        masks_dir: Directory containing masks
        color_r, color_g, color_b: RGB color values (0-255)
        alpha: Transparency (0.0-1.0)
    
    Returns:
        JPEG image with mask overlay
    """
    try:
        # Validate color values
        if not all(0 <= c <= 255 for c in [color_r, color_g, color_b]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Color values must be between 0 and 255",
            )
        
        # Validate alpha
        if not 0.0 <= alpha <= 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Alpha must be between 0.0 and 1.0",
            )
        
        viz_frame = editor_controller.visualize_tracking_result(
            frame_idx=frame_idx,
            masks_dir=masks_dir,
            color=(color_r, color_g, color_b),
            alpha=alpha
        )
        
        # Convert PIL image to JPEG bytes
        img_byte_arr = io.BytesIO()
        viz_frame.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename=tracking_frame_{frame_idx:06d}.jpg"
            }
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    
    except IndexError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error visualizing tracking: {str(e)}",
        )


@router.get("/tracking-summary")
async def get_tracking_summary_endpoint(masks_dir: str):
    """
    Get summary statistics for all tracking masks.
    
    Args:
        masks_dir: Directory containing masks
    
    Returns:
        JSON with summary statistics
    """
    try:
        masks = editor_controller.get_all_tracking_masks(masks_dir)
        
        # Calculate statistics
        total_frames = len(masks)
        frames_with_mask = sum(1 for m in masks if m.any())
        avg_coverage = sum(m.sum() / m.size for m in masks) / total_frames if total_frames > 0 else 0
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "data": {
                    "total_frames": total_frames,
                    "frames_with_mask": frames_with_mask,
                    "frames_without_mask": total_frames - frames_with_mask,
                    "average_mask_coverage": float(avg_coverage),
                    "masks_directory": masks_dir,
                }
            }
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error getting tracking summary: {str(e)}",
        )