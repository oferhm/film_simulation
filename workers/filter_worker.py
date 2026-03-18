"""
workers/filter_worker.py - QThread workers for off-thread filter processing.
"""

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from processing.filters import apply_filter_by_path


class FilterWorker(QThread):
    """Apply a film filter on a background thread."""

    # Emits (filtered_image, filter_name)
    filterApplied = pyqtSignal(np.ndarray, str)

    def __init__(self, image: np.ndarray, filter_path: str, filter_name: str):
        super().__init__()
        self.image = image
        self.filter_path = filter_path
        self.filter_name = filter_name

    def run(self):
        try:
            result = apply_filter_by_path(self.image, self.filter_path)
            self.filterApplied.emit(result, self.filter_name)
        except Exception as e:
            print(f"[FilterWorker] error applying '{self.filter_name}': {e}")


class FilterWorkerCached(QThread):
    """Apply a film filter and emit a cache key alongside the result."""

    # Emits (filtered_image, filter_name, cache_key)
    filterApplied = pyqtSignal(np.ndarray, str, str)

    def __init__(
        self,
        image: np.ndarray,
        filter_path: str,
        filter_name: str,
        cache_key: str,
    ):
        super().__init__()
        self.image = image
        self.filter_path = filter_path
        self.filter_name = filter_name
        self.cache_key = cache_key

    def run(self):
        try:
            result = apply_filter_by_path(self.image, self.filter_path)
            self.filterApplied.emit(result, self.filter_name, self.cache_key)
        except Exception as e:
            print(f"[FilterWorkerCached] error applying '{self.filter_name}': {e}")
