"""
ui/image_panel.py - Image display area with zoom/pan and comparison toggle.
"""

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import os
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QPushButton
)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QCursor


class ImagePanel(QFrame):
    """
    Displays the current image. Supports:
    - fit-to-window vs 100% zoom (left-click toggle)
    - pan when zoomed (drag)
    - comparison toggle button

    Signals
    -------
    toggleComparison()
    """

    toggleComparison = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imagePanel")
        self.setStyleSheet("""
            #imagePanel {
                background-color: #1e1e1e;
                border: 1px solid #3e3e42;
                border-radius: 0px;
            }
        """)

        self._pixmap: QPixmap | None = None
        self._is_zoomed  = False
        self._dragging   = False
        self._last_pan   = QPoint()
        self._click_start = QPoint()
        self._zoom_center: QPoint | None = None

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setStyleSheet("""
            QScrollArea {
                background-color: #1e1e1e;
                border: 1px solid #3e3e42;
                border-radius: 0px;
            }
            QScrollBar:vertical {
                background-color: #252526;
                border: none;
                width: 14px;
            }
            QScrollBar::handle:vertical {
                background-color: #424242;
                border-radius: 0px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4f4f4f;
            }
        """)
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.horizontalScrollBar().setSingleStep(1)
        self._scroll.verticalScrollBar().setSingleStep(1)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setMinimumSize(750, 500)
        self._img_label.setStyleSheet("background-color: transparent;")
        self._img_label.setMouseTracking(True)
        self._img_label.mousePressEvent   = self._on_press
        self._img_label.mouseMoveEvent    = self._on_move
        self._img_label.mouseReleaseEvent = self._on_release

        self._scroll.setWidget(self._img_label)
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll)

        # Info labels
        self._name_label = QLabel("No image loaded")
        self._name_label.setAlignment(Qt.AlignCenter)
        self._name_label.setStyleSheet(
            "background:transparent; color:#cccccc; font-size:12px; padding:8px; border:none;"
        )
        layout.addWidget(self._name_label)

        # Metadata labels
        self._metadata_frame = QFrame()
        self._metadata_frame.setStyleSheet(
            "background:transparent; border:none; padding:4px 0px;"
        )
        metadata_layout = QHBoxLayout(self._metadata_frame)
        metadata_layout.setContentsMargins(0, 0, 0, 0)
        metadata_layout.setSpacing(20)
        metadata_layout.addStretch()
        
        self._aperture_label = QLabel("")
        self._aperture_label.setAlignment(Qt.AlignCenter)
        self._aperture_label.setStyleSheet(
            "background:transparent; color:#999999; font-size:11px; border:none;"
        )
        metadata_layout.addWidget(self._aperture_label)
        
        self._shutter_label = QLabel("")
        self._shutter_label.setAlignment(Qt.AlignCenter)
        self._shutter_label.setStyleSheet(
            "background:transparent; color:#999999; font-size:11px; border:none;"
        )
        metadata_layout.addWidget(self._shutter_label)
        
        self._iso_label = QLabel("")
        self._iso_label.setAlignment(Qt.AlignCenter)
        self._iso_label.setStyleSheet(
            "background:transparent; color:#999999; font-size:11px; border:none;"
        )
        metadata_layout.addWidget(self._iso_label)
        
        self._camera_label = QLabel("")
        self._camera_label.setAlignment(Qt.AlignCenter)
        self._camera_label.setStyleSheet(
            "background:transparent; color:#999999; font-size:11px; border:none;"
        )
        metadata_layout.addWidget(self._camera_label)
        
        metadata_layout.addStretch()
        layout.addWidget(self._metadata_frame)

        # Bottom toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 0px;
                padding: 8px;
            }
        """)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(12, 8, 12, 8)
        tb_layout.addStretch()

        self._toggle_btn = QPushButton("Edited")
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #383838;
                border: 1px solid #464647;
                color: #cccccc;
                font-size: 11px;
                font-weight: 400;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover { 
                background-color: #2a2d2e; 
                border-color: #464647;
            }
            QPushButton:disabled {
                background-color: #252526;
                color: #656565;
                border-color: #3e3e42;
            }
        """)
        self._toggle_btn.clicked.connect(self.toggleComparison)
        tb_layout.addWidget(self._toggle_btn)
        layout.addWidget(toolbar)

    # ── public API ───────────────────────────────────────────────────────────

    def show_image(self, cv_image: np.ndarray, preserve_zoom: bool = False, image_path: str = None):
        """Display a BGR numpy array. Returns the created QPixmap for caching."""
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self._pixmap = pixmap

        was_zoomed = self._is_zoomed
        if preserve_zoom and was_zoomed:
            self._display_100(pixmap)
        else:
            self._display_fit(pixmap)
        
        # Update metadata if image path is provided
        if image_path:
            self._update_metadata(image_path)
        
        return pixmap  # Return for caching

    def show_pixmap(self, pixmap: QPixmap, preserve_zoom: bool = False):
        """Display a pre-converted QPixmap directly (faster for cached images)."""
        self._pixmap = pixmap
        
        was_zoomed = self._is_zoomed
        if preserve_zoom and was_zoomed:
            self._display_100(pixmap)
        else:
            self._display_fit(pixmap)

    def set_info(self, text: str):
        self._name_label.setText(text)

    def set_toggle_enabled(self, enabled: bool):
        self._toggle_btn.setEnabled(enabled)

    def set_toggle_text(self, text: str):
        self._toggle_btn.setText(text)

    def reset_zoom(self):
        self._is_zoomed = False
        self._dragging = False

    def _update_metadata(self, image_path: str):
        """Extract and display EXIF metadata from image."""
        try:
            if not os.path.exists(image_path):
                self._clear_metadata()
                return
            
            # Extract EXIF data
            with Image.open(image_path) as img:
                exif_dict = img._getexif()
                
            if exif_dict is not None:
                aperture_text = ""
                shutter_text = ""
                iso_text = ""
                camera_text = ""
                
                # Look for aperture (F-number)
                if 33437 in exif_dict:  # FNumber
                    f_num = exif_dict[33437]
                    if isinstance(f_num, tuple):
                        aperture = f_num[0] / f_num[1]
                    else:
                        aperture = float(f_num)
                    aperture_text = f"A: f/{aperture:.1f}"
                
                # Look for shutter speed (exposure time)
                if 33434 in exif_dict:  # ExposureTime
                    exposure = exif_dict[33434]
                    if isinstance(exposure, tuple):
                        if exposure[0] == 1:
                            shutter_text = f"S: 1/{exposure[1]}"
                        else:
                            shutter_speed = float(exposure[0]) / float(exposure[1])
                            if shutter_speed >= 1:
                                shutter_text = f"S: {shutter_speed:.1f}s"
                            else:
                                shutter_text = f"S: 1/{int(1/shutter_speed)}"
                    else:
                        exposure_val = float(exposure)
                        if exposure_val >= 1:
                            shutter_text = f"S: {exposure_val:.1f}s"
                        else:
                            shutter_text = f"S: 1/{int(1/exposure_val)}"
                
                # Look for ISO
                if 34855 in exif_dict:  # ISOSpeedRatings
                    iso = exif_dict[34855]
                    if isinstance(iso, (list, tuple)):
                        iso = iso[0]
                    iso_text = f"ISO: {int(iso)}"
                
                # Look for Camera (Make + Model)
                camera_make = ""
                camera_model = ""
                if 271 in exif_dict:  # Make
                    camera_make = str(exif_dict[271]).strip()
                if 272 in exif_dict:  # Model
                    camera_model = str(exif_dict[272]).strip()
                
                if camera_make and camera_model:
                    # Remove duplicate make from model if present
                    if camera_make.lower() in camera_model.lower():
                        camera_text = camera_model
                    else:
                        camera_text = f"{camera_make} {camera_model}"
                elif camera_model:
                    camera_text = camera_model
                elif camera_make:
                    camera_text = camera_make
                
                self._aperture_label.setText(aperture_text)
                self._shutter_label.setText(shutter_text)
                self._iso_label.setText(iso_text)
                self._camera_label.setText(camera_text)
            else:
                self._clear_metadata()
                
        except Exception as e:
            print(f"Error reading EXIF data: {e}")
            self._clear_metadata()
    
    def _clear_metadata(self):
        """Clear metadata display."""
        self._aperture_label.setText("")
        self._shutter_label.setText("")
        self._iso_label.setText("")
        self._camera_label.setText("")

    # ── zoom / pan ───────────────────────────────────────────────────────────

    def _display_fit(self, pixmap: QPixmap):
        self._scroll.setWidgetResizable(True)
        scaled = pixmap.scaled(self._img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img_label.setPixmap(scaled)
        self._img_label.setCursor(QCursor(Qt.ArrowCursor))

    def _display_100(self, pixmap: QPixmap):
        self._scroll.setWidgetResizable(False)
        self._img_label.resize(pixmap.size())
        self._img_label.setPixmap(pixmap)
        self._img_label.setCursor(QCursor(Qt.OpenHandCursor))

    def _toggle_zoom(self):
        if self._pixmap is None:
            return
        if self._is_zoomed:
            self._is_zoomed = False
            self._display_fit(self._pixmap)
        else:
            self._is_zoomed = True
            self._display_100(self._pixmap)
            if self._zoom_center:
                QTimer.singleShot(10, self._center_on_point)

    def _center_on_point(self):
        if not self._zoom_center:
            return
        vp = self._scroll.viewport().size()
        self._scroll.horizontalScrollBar().setValue(self._zoom_center.x() - vp.width() // 2)
        self._scroll.verticalScrollBar().setValue(self._zoom_center.y() - vp.height() // 2)

    def _calc_zoom_center(self, click_pos: QPoint):
        if not self._pixmap:
            return
        label = self._img_label.rect()
        pm = self._img_label.pixmap()
        if not pm:
            return
        sx = pm.width()  / self._pixmap.width()
        sy = pm.height() / self._pixmap.height()
        scale = min(sx, sy)
        x_off = (label.width()  - int(self._pixmap.width()  * scale)) // 2
        y_off = (label.height() - int(self._pixmap.height() * scale)) // 2
        ix = (click_pos.x() - x_off) / scale
        iy = (click_pos.y() - y_off) / scale
        self._zoom_center = QPoint(
            int(max(0, min(ix, self._pixmap.width()))),
            int(max(0, min(iy, self._pixmap.height()))),
        )

    # ── mouse events ─────────────────────────────────────────────────────────

    def _on_press(self, event):
        if self._pixmap is None:
            return
        if event.button() == Qt.LeftButton:
            self._click_start = event.pos()
            if not self._is_zoomed:
                self._calc_zoom_center(event.pos())
            else:
                self._last_pan = event.pos()

    def _on_move(self, event):
        if event.buttons() != Qt.LeftButton or not self._is_zoomed:
            return
        if not self._dragging:
            if (event.pos() - self._click_start).manhattanLength() > 3:
                self._dragging = True
                self._img_label.setCursor(QCursor(Qt.ClosedHandCursor))
        if self._dragging:
            delta = event.pos() - self._last_pan
            self._scroll.horizontalScrollBar().setValue(
                self._scroll.horizontalScrollBar().value() - delta.x()
            )
            self._scroll.verticalScrollBar().setValue(
                self._scroll.verticalScrollBar().value() - delta.y()
            )
        self._last_pan = event.pos()

    def _on_release(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self._dragging:
            self._dragging = False
            self._img_label.setCursor(
                QCursor(Qt.OpenHandCursor if self._is_zoomed else Qt.ArrowCursor)
            )
        else:
            if (event.pos() - self._click_start).manhattanLength() <= 5:
                self._toggle_zoom()
