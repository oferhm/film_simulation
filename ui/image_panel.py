"""
ui/image_panel.py - Image display area with zoom/pan and comparison toggle.
"""

import cv2
import numpy as np
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
                background-color: #1a1a1a;
                border: 2px solid #333333;
                border-radius: 12px;
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
                background-color: #0d0d0d;
                border: 2px solid #444444;
                border-radius: 8px;
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
            "background:transparent; color:#aaaaaa; font-size:12px; padding:8px; border:none;"
        )
        layout.addWidget(self._name_label)

        # Bottom toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 8px;
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
                background-color: #4a4a4a;
                border: 1px solid #666666;
                color: white;
                font-size: 11px;
                font-weight: 500;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #5a5a5a; }
        """)
        self._toggle_btn.clicked.connect(self.toggleComparison)
        tb_layout.addWidget(self._toggle_btn)
        layout.addWidget(toolbar)

    # ── public API ───────────────────────────────────────────────────────────

    def show_image(self, cv_image: np.ndarray, preserve_zoom: bool = False):
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
