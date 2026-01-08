from typing import Optional, List, Dict
from pydantic import BaseModel, Field, model_validator

class PairsRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)
    cursor: Optional[str] = None
    user_id: str = "default"

class InteractionItem(BaseModel):
    pairId: str
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    seen: Optional[bool] = None
    starred: Optional[bool] = None
    userId: str = "default"

class Circle(BaseModel):
    # accept a simple JSON array for center and validate ranges
    center: List[float] = Field(..., min_length=2, max_length=2, description="[lng, lat]")
    radius_m: float = Field(..., gt=0, description="Radius in meters")

    @model_validator(mode="after")
    def _check_center_ranges(self):
        lng, lat = self.center[0], self.center[1]
        if not (-180 <= lng <= 180 and -90 <= lat <= 90):
            raise ValueError("center must be [lng, lat] within valid ranges")
        return self

class PlotRequest(BaseModel):
    city:str
    inner_buffer: Optional[float] = Field(None, ge=0)
    outer_buffer: Optional[float] = Field(None, ge=0)
    area: Optional[Circle] = None

    @model_validator(mode="after")
    def _normalize_range(self):
        # if both provided and swapped, fix them (no error)
        if self.inner_buffer is not None and self.outer_buffer is not None:
            if self.inner_buffer > self.outer_buffer:
                self.inner_buffer, self.outer_buffer = self.outer_buffer, self.inner_buffer
        return self

class DownloadRequest(BaseModel):
    limit_pairs: int = 50
    cursor: Optional[str] = None
    all: bool = False
    include_locations: bool = False

class LikeExportRequest(PlotRequest):
    likes: List[Dict]