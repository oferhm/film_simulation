"""
core/state.py - Immutable edit state for all slider adjustments.
"""

from dataclasses import dataclass, replace
from typing import Optional
import numpy as np

try:
    from PyQt5.QtGui import QPixmap
except ImportError:
    QPixmap = None


@dataclass
class EditState:
    """Holds all current slider adjustment values."""
    temp:       int = 0
    tint:       int = 0
    exposure:   int = 0
    contrast:   int = 0
    highlights: int = 0
    shadows:    int = 0
    whites:     int = 0
    blacks:     int = 0
    texture:    int = 0
    clarity:    int = 0
    dehaze:     int = 0
    vibrance:   int = 0
    saturation: int = 0

    def is_default(self) -> bool:
        return all(
            getattr(self, f) == 0
            for f in self.__dataclass_fields__
        )

    def with_field(self, field: str, value: int) -> "EditState":
        """Return a new EditState with one field updated."""
        return replace(self, **{field: value})

    def reset(self) -> "EditState":
        return EditState()
    
    def __hash__(self) -> int:
        """Create hash for caching purposes."""
        return hash(tuple(getattr(self, f) for f in self.__dataclass_fields__))


@dataclass
class AppState:
    """Top-level application state."""
    original_image:  Optional[np.ndarray] = None
    filtered_image:  Optional[np.ndarray] = None
    selected_filter: Optional[str]        = None
    original_filename: Optional[str]      = None
    display_mode:    str                  = "original"   # "original" | "filtered"
    is_zoomed:       bool                 = False
    edits:           EditState            = None
    # Cache for processed image and QPixmaps to avoid recomputation
    _cached_processed_image: Optional[np.ndarray] = None
    _cached_edits_hash: Optional[int] = None
    _cached_original_pixmap: Optional = None  # QPixmap cache for original
    _cached_processed_pixmap: Optional = None  # QPixmap cache for processed
    # Cache for full-resolution filter results
    _cached_filters: Optional[dict] = None  # {filter_name: filtered_image}

    def __post_init__(self):
        if self.edits is None:
            self.edits = EditState()
        if self._cached_filters is None:
            self._cached_filters = {}

    @property
    def has_edits(self) -> bool:
        return (
            self.filtered_image is not None
            or not self.edits.is_default()
        )

    def reset_for_new_image(self) -> "AppState":
        return AppState(
            original_image=self.original_image,
            original_filename=self.original_filename,
        )
