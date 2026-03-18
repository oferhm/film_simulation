"""
ui/main_window.py - QMainWindow: wires toolbar, image panel, controls panel.

All image-processing logic lives in processing/. All state lives in core/state.
This file only connects signals, delegates work, and updates the UI.
"""

import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QShortcut
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

from core.state import AppState, EditState
from core.image_io import load_image, save_image, next_available_filename
from processing.adjustments import apply_all
from processing.filters import apply_filter_by_path
from workers.filter_worker import FilterWorker, FilterWorkerCached
from ui.toolbar import TopToolbar
from ui.image_panel import ImagePanel
from ui.controls_panel import ControlsPanel

try:
    import dynamic_config
except ImportError:
    class dynamic_config:
        import_folder_path = ""
        export_folder_path = ""


_PREVIEW_MAX_PX = 800   # longest side used for hover previews


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._state = AppState()
        self._filter_worker: FilterWorker | None = None
        self._preview_cache: dict[str, np.ndarray] = {}

        self._build_ui()
        self._connect_signals()
        self._apply_window_style()

        # Shortcut for toggle comparison  
        QShortcut(QKeySequence("\\"), self).activated.connect(
            self._on_toggle_comparison
        )

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Film Filter Studio")
        self.setGeometry(100, 100, 1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._toolbar  = TopToolbar()
        self._imgpanel = ImagePanel()
        self._controls = ControlsPanel(filters_dir="film_filters/filters")

        root.addWidget(self._toolbar)

        content = QHBoxLayout()
        content.setSpacing(8)
        content.addWidget(self._imgpanel, 4)
        content.addWidget(self._controls, 1)
        root.addLayout(content)

        # Sync path labels
        self._toolbar.set_import_path(getattr(dynamic_config, "import_folder_path", ""))
        self._toolbar.set_save_path(getattr(dynamic_config, "export_folder_path", ""))

    def _connect_signals(self):
        self._toolbar.importClicked.connect(self._on_import)
        self._toolbar.saveClicked.connect(self._on_save)
        self._imgpanel.toggleComparison.connect(self._on_toggle_comparison)
        self._controls.editChanged.connect(self._on_edit_changed)
        self._controls.filterSelected.connect(self._on_filter_selected)
        self._controls.filterDeselected.connect(self._on_filter_deselected)
        self._controls.filterHovered.connect(self._on_filter_hover)
        self._controls.filterHoverLeft.connect(self._on_filter_hover_left)

    # ── Import ───────────────────────────────────────────────────────────────

    def _on_import(self):
        start_dir = getattr(dynamic_config, "import_folder_path", "") or "/"
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", start_dir,
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
        )
        if not path:
            return

        img = load_image(path)
        if img is None:
            QMessageBox.critical(self, "Error", f"Could not load:\n{path}")
            return

        # Persist import folder
        folder = os.path.dirname(path)
        self._update_dynamic_config("import_folder_path", folder)
        self._toolbar.set_import_path(folder)

        # Reset state
        self._state = AppState(
            original_image=img,
            original_filename=os.path.basename(path),
            # Cache starts clean for new image
            _cached_processed_image=None,
            _cached_edits_hash=None,
            _cached_original_pixmap=None,
            _cached_processed_pixmap=None,
            _cached_filters={},  # Fresh filter cache for new image
        )
        self._preview_cache.clear()
        self._controls.reset_all()
        self._controls.deselect_filter()
        self._imgpanel.reset_zoom()

        # Display
        self._imgpanel.show_image(img)
        self._imgpanel.set_info(self._build_info_text(path, img))
        self._imgpanel.set_toggle_enabled(False)
        self._imgpanel.set_toggle_text("Original")
        self._toolbar.set_save_enabled(True)
        self._controls.set_controls_enabled(True)

    # ── Save ─────────────────────────────────────────────────────────────────

    def _on_save(self):
        if self._state.original_image is None:
            return

        start_dir = getattr(dynamic_config, "export_folder_path", "") or "/"
        base, ext = os.path.splitext(self._state.original_filename or "image")
        ext = ext.lower() if ext.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff") else ".jpg"
        suggested = next_available_filename(start_dir, base, ext)
        default_path = os.path.join(start_dir, suggested)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image As", default_path,
            "JPEG (*.jpg);;PNG (*.png);;All Images (*.jpg *.png *.bmp *.tiff)"
        )
        if not path:
            return

        folder = os.path.dirname(path)
        self._update_dynamic_config("export_folder_path", folder)
        self._toolbar.set_save_path(folder)

        image_to_save = self._build_final_image()
        if not save_image(path, image_to_save):
            QMessageBox.critical(self, "Error", "Failed to save image.")

    # ── Edit changes (sliders) ───────────────────────────────────────────────

    def _on_edit_changed(self, edit_state: EditState):
        self._state = AppState(
            original_image=self._state.original_image,
            filtered_image=self._state.filtered_image,
            selected_filter=self._state.selected_filter,
            original_filename=self._state.original_filename,
            display_mode="filtered" if not edit_state.is_default() or self._state.filtered_image is not None else "original",
            is_zoomed=self._state.is_zoomed,
            edits=edit_state,
            # Keep numpy cache, clear only if edits changed significantly 
            _cached_processed_image=self._state._cached_processed_image,
            _cached_edits_hash=self._state._cached_edits_hash,
            _cached_original_pixmap=self._state._cached_original_pixmap,  # Keep original pixmap
            _cached_processed_pixmap=None,  # Clear processed pixmap - will be recreated if needed
            _cached_filters=self._state._cached_filters,  # Keep filter cache
        )
        self._refresh_display()
        self._sync_toggle_button()

    # ── Filter selection ─────────────────────────────────────────────────────

    def _on_filter_selected(self, name: str, path: str):
        if self._state.original_image is None:
            return
            
        # Check if we have this filter cached
        if name in self._state._cached_filters:
            # Use cached result - instant!
            cached_result = self._state._cached_filters[name]
            self._on_filter_applied(cached_result, name)
        else:
            # Process filter and cache result
            self._run_filter(self._state.original_image, path, name, is_preview=False)

    def _on_filter_deselected(self):
        self._state = AppState(
            original_image=self._state.original_image,
            original_filename=self._state.original_filename,
            display_mode="original" if self._state.edits.is_default() else "filtered",
            edits=self._state.edits,
            # Preserve all caches when deselecting
            _cached_processed_image=self._state._cached_processed_image,
            _cached_edits_hash=self._state._cached_edits_hash,
            _cached_original_pixmap=self._state._cached_original_pixmap,
            _cached_processed_pixmap=self._state._cached_processed_pixmap,
            _cached_filters=self._state._cached_filters,
        )
        self._refresh_display()
        self._sync_toggle_button()

    def _on_filter_applied(self, filtered: np.ndarray, name: str):
        # Cache the full-resolution filter result
        self._state._cached_filters[name] = filtered.copy()
        
        self._state = AppState(
            original_image=self._state.original_image,
            filtered_image=filtered,
            selected_filter=name,
            original_filename=self._state.original_filename,
            display_mode="filtered",
            edits=self._state.edits,
            # Invalidate adjustment cache since base image changed
            _cached_processed_image=None,
            _cached_edits_hash=None,
            _cached_original_pixmap=self._state._cached_original_pixmap,  # Keep original pixmap
            _cached_processed_pixmap=None,  # Clear processed pixmap
            _cached_filters=self._state._cached_filters,  # Preserve filter cache
        )
        self._refresh_display()
        self._sync_toggle_button()
        self._controls.set_controls_enabled(True)

    # ── Hover preview ────────────────────────────────────────────────────────

    def _on_filter_hover(self, name: str, path: str):
        if self._state.original_image is None:
            return
        if self._state.selected_filter is not None:
            return   # don't preview when a filter is locked in

        cache_key = f"{name}_exp{self._state.edits.exposure}_shd{self._state.edits.shadows}"
        if cache_key in self._preview_cache:
            self._imgpanel.show_image(self._preview_cache[cache_key], preserve_zoom=True)
            return

        # Downscale for speed
        base = self._downscale(self._state.original_image, _PREVIEW_MAX_PX)
        base = apply_all(base, self._state.edits)
        self._run_filter(base, path, name, is_preview=True, cache_key=cache_key)

    def _on_filter_hover_left(self):
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(25, self._restore_current_view)

    def _on_preview_ready(self, preview: np.ndarray, name: str, cache_key: str):
        self._preview_cache[cache_key] = preview
        self._imgpanel.show_image(preview, preserve_zoom=True)

    # ── Comparison toggle ────────────────────────────────────────────────────

    def _on_toggle_comparison(self):
        if self._state.original_image is None:
            return
        if self._state.display_mode == "original":
            if self._state.has_edits:
                self._state = AppState(
                    original_image=self._state.original_image,
                    filtered_image=self._state.filtered_image,
                    selected_filter=self._state.selected_filter,
                    original_filename=self._state.original_filename,
                    display_mode="filtered",
                    edits=self._state.edits,
                    # Preserve all caches during toggle
                    _cached_processed_image=self._state._cached_processed_image,
                    _cached_edits_hash=self._state._cached_edits_hash,
                    _cached_original_pixmap=self._state._cached_original_pixmap,
                    _cached_processed_pixmap=self._state._cached_processed_pixmap,
                    _cached_filters=self._state._cached_filters,
                )
        else:
            self._state = AppState(
                original_image=self._state.original_image,
                filtered_image=self._state.filtered_image,
                selected_filter=self._state.selected_filter,
                original_filename=self._state.original_filename,
                display_mode="original",
                edits=self._state.edits,
                # Preserve all caches during toggle
                _cached_processed_image=self._state._cached_processed_image,
                _cached_edits_hash=self._state._cached_edits_hash,
                _cached_original_pixmap=self._state._cached_original_pixmap,
                _cached_processed_pixmap=self._state._cached_processed_pixmap,
            )
        self._refresh_display()
        self._sync_toggle_button()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _refresh_display(self):
        if self._state.original_image is None:
            return
        if self._state.display_mode == "original":
            # Use cached QPixmap for original if available
            if self._state._cached_original_pixmap is not None:
                self._imgpanel.show_pixmap(self._state._cached_original_pixmap, preserve_zoom=True)
            else:
                pixmap = self._imgpanel.show_image(self._state.original_image, preserve_zoom=True)
                self._state._cached_original_pixmap = pixmap
        else:
            # Use cached QPixmap for processed if available
            current_edits_hash = hash(self._state.edits)
            if (self._state._cached_processed_pixmap is not None and 
                self._state._cached_edits_hash == current_edits_hash):
                self._imgpanel.show_pixmap(self._state._cached_processed_pixmap, preserve_zoom=True)
            else:
                final_image = self._build_final_image()
                pixmap = self._imgpanel.show_image(final_image, preserve_zoom=True)
                self._state._cached_processed_pixmap = pixmap
                self._state._cached_edits_hash = current_edits_hash

    def _restore_current_view(self):
        if self._state.original_image is None:
            return
        self._refresh_display()

    def _build_final_image(self) -> np.ndarray:
        # Check if we can use cached processed image
        current_edits_hash = hash(self._state.edits)
        if (self._state._cached_processed_image is not None and 
            self._state._cached_edits_hash == current_edits_hash):
            return self._state._cached_processed_image
        
        # Need to recompute
        base = self._state.filtered_image if self._state.filtered_image is not None \
               else self._state.original_image
        result = apply_all(base.copy(), self._state.edits)
        
        # Cache the result
        self._state._cached_processed_image = result
        self._state._cached_edits_hash = current_edits_hash
        
        return result

    def _run_filter(
        self, image: np.ndarray, path: str, name: str,
        is_preview: bool, cache_key: str = ""
    ):
        if self._filter_worker and self._filter_worker.isRunning():
            return
        if is_preview:
            worker = FilterWorkerCached(image, path, name, cache_key)
            worker.filterApplied.connect(self._on_preview_ready)
        else:
            self._controls.set_controls_enabled(False)
            worker = FilterWorker(image, path, name)
            worker.filterApplied.connect(self._on_filter_applied)
        self._filter_worker = worker
        worker.start()

    def _sync_toggle_button(self):
        has = self._state.has_edits
        self._imgpanel.set_toggle_enabled(has)
        if not has:
            self._imgpanel.set_toggle_text("Original")
        elif self._state.display_mode == "original":
            self._imgpanel.set_toggle_text("Original")
        else:
            self._imgpanel.set_toggle_text("Edited")

    @staticmethod
    def _downscale(image: np.ndarray, max_px: int) -> np.ndarray:
        import cv2
        h, w = image.shape[:2]
        if max(h, w) <= max_px:
            return image
        scale = max_px / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _build_info_text(path: str, img: np.ndarray) -> str:
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            h, w = img.shape[:2]
            from math import gcd
            g = gcd(w, h)
            return f"{os.path.basename(path)}  •  {w} × {h}  •  {w//g}:{h//g}  •  {size_mb:.1f} MB"
        except Exception:
            return os.path.basename(path)

    @staticmethod
    def _update_dynamic_config(key: str, value: str):
        try:
            cfg_file = "dynamic_config.py"
            with open(cfg_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(cfg_file, "w", encoding="utf-8") as f:
                for line in lines:
                    if line.strip().startswith(f"{key} ="):
                        f.write(f'{key} = "{value}"\n')
                    else:
                        f.write(line)
            setattr(dynamic_config, key, value)
        except Exception as e:
            print(f"[dynamic_config] could not update {key}: {e}")

    # ── Window style ─────────────────────────────────────────────────────────

    def _apply_window_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                font-family: "Segoe UI", "Arial", sans-serif;
            }
            QFrame {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 0px;
                margin: 2px;
            }
            QLabel  { color: #cccccc; font-size: 12px; }
            QScrollArea {
                border: 1px solid #3e3e42;
                background-color: #252526;
                border-radius: 0px;
            }
            QScrollBar:vertical {
                background: #252526; 
                width: 14px; 
                border-radius: 0px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #424242; 
                border-radius: 0px; 
                min-height: 20px; 
                margin: 0px;
            }
            QScrollBar::handle:vertical:hover  { background: #4f4f4f; }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                background: #252526; 
                height: 0px; 
                border-radius: 0px;
            }
            QScrollBar:horizontal {
                background: #252526; 
                height: 14px; 
                border-radius: 0px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #424242; 
                border-radius: 0px; 
                min-width: 20px; 
                margin: 0px;
            }
            QScrollBar::handle:horizontal:hover { background: #4f4f4f; }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                background: #252526; 
                width: 0px; 
                border-radius: 0px;
            }
        """)
