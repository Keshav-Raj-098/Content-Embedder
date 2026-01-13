from pydantic import BaseModel,Field,field_validator
from typing import Annotated, List


class LoadVideoRequest(BaseModel):
    video_path: str



Point = Annotated[List[float], Field(min_length=2, max_length=2)]
Quad = Annotated[List[Point], Field(min_length=4, max_length=4)]

class PlacementRequest(BaseModel):
    quad: Quad
    
    @field_validator("quad")
    @classmethod
    def validate_numbers(cls, quad):
        for p in quad:
            if len(p) != 2:
                raise ValueError("Each point must be [x, y]")
        return quad