#!/usr/bin/env python3
"""
Film Filter GUI Application

A PyQt5 GUI application for applying various film filters to photos in real-time.
Features:
- Load and display photos
- Apply filters with live preview
- Original/Filtered comparison
- Support for all available film filters
"""

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QScrollArea, 
                            QFileDialog, QMessageBox, QFrame, QSizePolicy, QShortcut, 
                            QSlider, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt5.QtGui import QPixmap, QImage, QFont, QCursor, QKeySequence
import importlib.util
from pathlib import Path

# Import config for default paths
try:
    import config
except ImportError:
    # Fallback if config not found
    class config:
        upload_photo_path = ""

# Import dynamic config for persistent paths
try:
    import dynamic_config
except ImportError:
    # Fallback if dynamic_config not found
    class dynamic_config:
        import_folder_path = ""
        export_folder_path = ""


class FilterButton(QPushButton):
    """Custom button with hover preview functionality and selection state."""
    hoverEntered = pyqtSignal(str, str)  # filter_name, filter_path
    hoverLeft = pyqtSignal()
    
    def __init__(self, filter_name, filter_path, parent=None):
        super().__init__(filter_name, parent)
        self.filter_name = filter_name
        self.filter_path = filter_path
        self.setFixedHeight(45)
        self.is_selected = False
        
        # Add hover delay to prevent flickering
        self.hover_timer = QTimer()
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(self._emit_hover_entered)
        self.hover_delay = 50  # Very fast response for smooth transitions
        
    def set_selected(self, selected):
        """Set the selected state and update styling."""
        self.is_selected = selected
        self.update_style()
        
    def update_style(self):
        """Update button styling based on selection state."""
        if self.is_selected:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    border: 2px solid #0078d4;
                    border-radius: 8px;
                    padding: 15px 16px;
                    font-size: 13px;
                    font-weight: 600;
                    color: #ffffff;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #4c4c4c;
                    border-color: #106ebe;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    border: 1px solid #555555;
                    border-radius: 8px;
                    padding: 16px;
                    font-size: 13px;
                    font-weight: 500;
                    color: #ffffff;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #4c4c4c;
                    border-color: #777777;
                }
                QPushButton:pressed {
                    background-color: #2c2c2c;
                }
                QPushButton:disabled {
                    background-color: #282828;
                    color: #666666;
                    border-color: #333333;
                }
            """)
        
    def enterEvent(self, event):
        """Handle mouse enter event with delay."""
        super().enterEvent(event)
        # Start timer for hover delay
        self.hover_timer.start(self.hover_delay)
        
    def leaveEvent(self, event):
        """Handle mouse leave event."""
        super().leaveEvent(event)
        # Cancel hover timer if still running
        if self.hover_timer.isActive():
            self.hover_timer.stop()
        else:
            # Only emit if hover was actually triggered
            self.hoverLeft.emit()
            
    def _emit_hover_entered(self):
        """Emit hover entered signal after delay."""
        self.hoverEntered.emit(self.filter_name, self.filter_path)


class FilterWorker(QThread):
    """Worker thread for applying filters to avoid GUI freezing."""
    filterApplied = pyqtSignal(np.ndarray, str)
    
    def __init__(self, image, filter_path, filter_name):
        super().__init__()
        self.image = image
        self.filter_path = filter_path
        self.filter_name = filter_name
    
    def run(self):
        try:
            # Apply the filter to the image
            filtered_image = self.apply_filter(self.image, self.filter_path)
            self.filterApplied.emit(filtered_image, self.filter_name)
        except Exception as e:
            print(f"Error applying filter {self.filter_name}: {e}")
    
    def apply_filter(self, image, filter_path):
        """
        Dynamically apply a filter by importing and executing its logic.
        """
        # Save original working directory
        original_cwd = os.getcwd()
        
        try:
            # Ensure image has 3 color channels for filter processing
            if len(image.shape) == 2:
                # Convert grayscale to BGR
                temp_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # Convert single channel to BGR
                temp_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                temp_image = image.copy()
            
            # Convert to float32 for processing
            temp_image = temp_image.astype(np.float32) / 255.0
            
            # Execute filter logic with the image
            filtered_image = self.execute_filter_logic(temp_image, filter_path)
            
            # Convert back to uint8
            filtered_image = np.clip(filtered_image * 255, 0, 255).astype(np.uint8)
            
            return filtered_image
            
        except Exception as e:
            print(f"Filter application error: {e}")
            return image  # Return original on error
        finally:
            # Restore working directory
            os.chdir(original_cwd)
    
    def execute_filter_logic(self, image, filter_path):
        """Extract and execute the core filter logic from filter files."""
        filter_name = os.path.basename(filter_path)
        
        if "kodak_portra" in filter_name.lower():
            return self.apply_kodak_portra_filter(image)
        elif "ektar" in filter_name.lower():
            return self.apply_ektar_filter(image)
        elif "vintage" in filter_name.lower():
            return self.apply_vintage_filter(image)
        elif "ilford" in filter_name.lower():
            return self.apply_ilford_filter(image)
        elif "colorplus" in filter_name.lower():
            if "bright" in filter_name.lower():
                return self.apply_colorplus_bright_filter(image)
            else:
                return self.apply_colorplus_filter(image)
        elif "gold" in filter_name.lower():
            return self.apply_gold_filter(image)
        elif "expired" in filter_name.lower():
            return self.apply_expired_filter(image)
        else:
            # Default processing for unknown filters
            return self.apply_generic_filter(image)
    
    def apply_s_curve(self, image, strength=1.1):
        """Apply S-curve for contrast enhancement."""
        curve = np.linspace(0, 1, 256)
        curve = np.clip((curve - 0.5) * strength + 0.5, 0, 1)
        
        # Ensure image has proper dimensions
        if len(image.shape) == 2:
            # Grayscale image
            return np.interp(image, np.linspace(0, 1, 256), curve)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Single channel 3D array
            return np.interp(image[:, :, 0], np.linspace(0, 1, 256), curve).reshape(image.shape[0], image.shape[1], 1)
        else:
            # Apply curve to each channel
            result = image.copy()
            for i in range(min(3, image.shape[2])):  # Handle up to 3 channels safely
                result[:, :, i] = np.interp(result[:, :, i], np.linspace(0, 1, 256), curve)
            return result
    
    def apply_kodak_portra_filter(self, image):
        """Apply Kodak Portra style filter."""
        # Ensure image has 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # S-curve for contrast
        result = self.apply_s_curve(image, 1.1)
        
        # Warm color grading
        result[:, :, 0] *= 0.95  # Reduce blue slightly
        result[:, :, 2] *= 1.05  # Increase red slightly
        
        # Slight saturation boost
        result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        result[:, :, 1] *= 1.1  # Increase saturation
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        
        return np.clip(result, 0, 1)
    
    def apply_ektar_filter(self, image):
        """Apply Ektar 100 style filter."""
        result = image.copy()
        
        # High saturation and contrast
        result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        result[:, :, 1] *= 1.3  # High saturation
        result[:, :, 2] *= 1.1  # Slight brightness boost
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        
        # Apply strong S-curve
        result = self.apply_s_curve(result, 1.3)
        
        return np.clip(result, 0, 1)
    
    def apply_vintage_filter(self, image):
        """Apply vintage style filter."""
        # Ensure BGR format
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        result = image.copy()
        
        # Sepia-like color grading
        result[:, :, 0] *= 0.8   # Reduce blue
        result[:, :, 1] *= 0.95  # Slightly reduce green
        result[:, :, 2] *= 1.1   # Increase red
        
        # Add slight vignette
        h, w = result.shape[:2]
        center_x, center_y = w // 2, h // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        vignette = 1 - (dist / max_dist) * 0.3
        
        result = result * vignette[:, :, np.newaxis]
        
        return np.clip(result, 0, 1)
    
    def apply_ilford_filter(self, image):
        """Apply Ilford HP5+ B&W style filter."""
        # Convert to grayscale with film-like response
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast curve
        gray = self.apply_s_curve(gray.reshape(gray.shape[0], gray.shape[1], 1), 1.2)
        
        # Convert back to BGR
        result = np.dstack([gray, gray, gray]).squeeze()
        
        return result
    
    def apply_colorplus_filter(self, image):
        """Apply Kodak ColorPlus 200 style filter."""
        # Ensure BGR format
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        result = image.copy()
        
        # Slightly muted colors
        result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        result[:, :, 1] *= 0.9  # Reduce saturation slightly
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        
        # Warm tone
        result[:, :, 0] *= 0.9   # Reduce blue
        result[:, :, 2] *= 1.05  # Slight red boost
        
        return np.clip(result, 0, 1)
    
    def apply_colorplus_bright_filter(self, image):
        """Apply bright version of ColorPlus filter."""
        result = self.apply_colorplus_filter(image)
        
        # Increase overall brightness
        result += 0.1
        
        return np.clip(result, 0, 1)
    
    def apply_gold_filter(self, image):
        """Apply Kodak Gold 200 style filter."""
        # Ensure BGR format
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        result = image.copy()
        
        # Golden color cast
        result[:, :, 0] *= 0.85  # Reduce blue
        result[:, :, 1] *= 1.05  # Slight green boost
        result[:, :, 2] *= 1.15  # Increase red for warmth
        
        return np.clip(result, 0, 1)
    
    def apply_expired_filter(self, image):
        """Apply expired film style filter."""
        # Ensure BGR format
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        result = image.copy()
        
        # Fade effect - lift shadows
        result += 0.03
        
        # Color shift
        result[:, :, 0] *= 0.98  # Slight blue reduction
        result[:, :, 1] *= 1.02  # Slight green boost
        
        # Soft vignette
        h, w = result.shape[:2]
        center_x, center_y = w // 2, h // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        vignette = 1 - (dist / max_dist) * 0.1  # Very subtle
        
        result = result * vignette[:, :, np.newaxis]
        
        return np.clip(result, 0, 1)
    
    def apply_generic_filter(self, image):
        """Apply generic enhancement for unknown filters."""
        return self.apply_s_curve(image, 1.05)


class FilterWorkerWithCacheKey(QThread):
    """Worker thread for applying filters with cache key support."""
    filterApplied = pyqtSignal(np.ndarray, str, str)  # image, filter_name, cache_key
    
    def __init__(self, image, filter_path, filter_name, cache_key):
        super().__init__()
        self.image = image
        self.filter_path = filter_path
        self.filter_name = filter_name
        self.cache_key = cache_key
    
    def run(self):
        try:
            # Apply the filter to the image using FilterWorker logic
            filter_worker = FilterWorker(self.image, self.filter_path, self.filter_name)
            filtered_image = filter_worker.apply_filter(self.image, self.filter_path)
            self.filterApplied.emit(filtered_image, self.filter_name, self.cache_key)
        except Exception as e:
            print(f"Error applying filter {self.filter_name}: {e}")


class FilmFilterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.current_filtered_image = None
        self.current_display_mode = "original"  # "original" or "filtered"
        self.filter_worker = None
        self.preview_cache = {}  # Cache for hover previews
        self.is_hovering = False  # Track hover state
        self.currently_hovering_filter = None  # Track which filter is being hovered
        self.selected_filter = None  # Track currently selected filter
        self.selected_filter_button = None  # Track selected button
        self.original_filename = None  # Track original filename for save suggestions
        self.is_zoomed = False  # Track zoom state
        self.original_pixmap = None  # Store original pixmap for zoom
        self.dragging = False  # Track drag state
        self.last_pan_point = QPoint()  # Last mouse position for dragging
        self.left_click_start_pos = QPoint()  # Track left-click start position
        self.zoom_center_point = None  # Point to center when zooming
        
        # Get available filters
        self.filters_dir = "film_filters/filters"
        self.available_filters = self.get_available_filters()
        
        # Initialize exposure adjustment
        self.exposure_adjustment = 0  # Range: -100 to +100
        
        # Initialize shadow adjustment  
        self.shadow_adjustment = 0  # Range: -100 to +100
        
        self.init_ui()
        
        # Add keyboard shortcut for toggle comparison
        self.toggle_shortcut = QShortcut(QKeySequence("\\"), self)
        self.toggle_shortcut.activated.connect(self.toggle_comparison)
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Film Filter Studio - Professional Photo Editing")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Top toolbar with Import button
        top_toolbar = self.create_top_toolbar()
        main_layout.addWidget(top_toolbar)
        
        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(8)
        
        # Left side - Image display (takes most space)
        image_panel = self.create_image_panel()
        content_layout.addWidget(image_panel, 3)  # 75% of space
        
        # Right side - Controls panel
        controls_panel = self.create_controls_panel()
        content_layout.addWidget(controls_panel, 1)  # 25% of space
        
        main_layout.addLayout(content_layout)
        
        # Set modern window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: "Segoe UI", "Arial", sans-serif;
            }
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 8px;
                margin: 2px;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 13px;
                font-weight: 500;
                color: #ffffff;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #666666;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
                border-color: #333333;
            }
            QPushButton:disabled {
                background-color: #262626;
                color: #666666;
                border-color: #333333;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QScrollArea {
                border: 1px solid #404040;
                background-color: #2d2d2d;
                border-radius: 6px;
            }
            /* Enhanced Scroll Bars - Vertical */
            QScrollBar:vertical {
                border: 1px solid #404040;
                background: #1e1e1e;
                width: 20px;
                border-radius: 10px;
                margin: 22px 0 22px 0;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #666666, stop:1 #555555);
                border: 1px solid #666666;
                border-radius: 8px;
                min-height: 30px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0078d4, stop:1 #106ebe);
                border-color: #106ebe;
            }
            QScrollBar::handle:vertical:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #005a9e, stop:1 #004578);
                border-color: #004578;
            }
            QScrollBar::add-line:vertical {
                border: 1px solid #404040;
                background: #2d2d2d;
                height: 20px;
                border-radius: 10px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }
            QScrollBar::sub-line:vertical {
                border: 1px solid #404040;
                background: #2d2d2d;
                height: 20px;
                border-radius: 10px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }
            QScrollBar::add-line:vertical:hover,
            QScrollBar::sub-line:vertical:hover {
                background: #3d3d3d;
                border-color: #555555;
            }
            QScrollBar::add-line:vertical:pressed,
            QScrollBar::sub-line:vertical:pressed {
                background: #1e1e1e;
            }
            QScrollBar:up-arrow:vertical, QScrollBar:down-arrow:vertical {
                width: 8px;
                height: 8px;
                background: #888888;
            }
            QScrollBar:up-arrow:vertical:hover, QScrollBar:down-arrow:vertical:hover {
                background: #aaaaaa;
            }
            
            /* Enhanced Scroll Bars - Horizontal */
            QScrollBar:horizontal {
                border: 1px solid #404040;
                background: #1e1e1e;
                height: 20px;
                border-radius: 10px;
                margin: 0 22px 0 22px;
            }
            QScrollBar::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #666666, stop:1 #555555);
                border: 1px solid #666666;
                border-radius: 8px;
                min-width: 30px;
                margin: 1px;
            }
            QScrollBar::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0078d4, stop:1 #106ebe);
                border-color: #106ebe;
            }
            QScrollBar::handle:horizontal:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #005a9e, stop:1 #004578);
                border-color: #004578;
            }
            QScrollBar::add-line:horizontal {
                border: 1px solid #404040;
                background: #2d2d2d;
                width: 20px;
                border-radius: 10px;
                subcontrol-position: right;
                subcontrol-origin: margin;
            }
            QScrollBar::sub-line:horizontal {
                border: 1px solid #404040;
                background: #2d2d2d;
                width: 20px;
                border-radius: 10px;
                subcontrol-position: left;
                subcontrol-origin: margin;
            }
            QScrollBar::add-line:horizontal:hover,
            QScrollBar::sub-line:horizontal:hover {
                background: #3d3d3d;
                border-color: #555555;
            }
            QScrollBar::add-line:horizontal:pressed,
            QScrollBar::sub-line:horizontal:pressed {
                background: #1e1e1e;
            }
            QScrollBar:left-arrow:horizontal, QScrollBar:right-arrow:horizontal {
                width: 8px;
                height: 8px;
                background: #888888;
            }
            QScrollBar:left-arrow:horizontal:hover, QScrollBar:right-arrow:horizontal:hover {
                background: #aaaaaa;
            }
            .title-label {
                font-size: 18px;
                font-weight: 600;
                color: #ffffff;
                padding: 16px;
                background-color: transparent;
                border: none;
            }
            .status-label {
                background-color: #333333;
                padding: 12px;
                border-radius: 6px;
                font-size: 11px;
                color: #cccccc;
                border: 1px solid #444444;
            }
            .main-button {
                background-color: #0078d4;
                border: 1px solid #106ebe;
                font-size: 14px;
                font-weight: 600;
                padding: 16px;
            }
            .main-button:hover {
                background-color: #106ebe;
                border-color: #005a9e;
            }
            .comparison-button {
                background-color: #4a4a4a;
                border: 1px solid #666666;
                font-size: 12px;
                padding: 10px 20px;
            }
            .comparison-button:hover {
                background-color: #5a5a5a;
            }
        """)
    
    def create_top_toolbar(self):
        """Create the top toolbar with Import button."""
        toolbar = QFrame()
        toolbar.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 8, 12, 8)
        
        # Import button (styled like filter buttons)
        self.load_btn = QPushButton("Import Photos")
        self.load_btn.setFixedHeight(36)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 500;
                color: #ffffff;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
                border-color: #777777;
            }
            QPushButton:pressed {
                background-color: #2c2c2c;
            }
        """)
        self.load_btn.clicked.connect(self.load_image)
        toolbar_layout.addWidget(self.load_btn)
        
        # Import path label
        self.import_path_label = QLabel()
        self.import_path_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 11px;
                padding: 4px 8px;
                background-color: transparent;
                max-width: 300px;
                min-width: 200px;
            }
        """)
        self.update_import_path_label()
        toolbar_layout.addWidget(self.import_path_label)
        
        # Add larger stretch to position save button more to the left
        toolbar_layout.addStretch(2)
        
        # Save button (styled like filter buttons)
        self.save_btn = QPushButton("Save as")
        self.save_btn.setFixedHeight(36)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 500;
                color: #ffffff;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
                border-color: #777777;
            }
            QPushButton:pressed {
                background-color: #2c2c2c;
            }
            QPushButton:disabled {
                background-color: #282828;
                color: #666666;
                border-color: #333333;
            }
        """)
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)  # Disabled until image is loaded
        toolbar_layout.addWidget(self.save_btn)
        
        # Save path label
        self.save_path_label = QLabel()
        self.save_path_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 11px;
                padding: 4px 8px;
                background-color: transparent;
                max-width: 300px;
                min-width: 200px;
            }
        """)
        self.update_save_path_label()
        toolbar_layout.addWidget(self.save_path_label)
        
        # Add larger stretch after save button
        toolbar_layout.addStretch(2)
        
        return toolbar
    
    def create_image_panel(self):
        """Create the modern image display panel."""
        panel = QFrame()
        panel.setObjectName("imagePanel")
        panel.setStyleSheet("""
            #imagePanel {
                background-color: #1a1a1a;
                border: 2px solid #333333;
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Image display area
        self.image_scroll = QScrollArea()
        self.image_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #0d0d0d;
                border: 2px solid #444444;
                border-radius: 8px;
            }
        """)
        # Configure scroll area for robust dragging with thick scroll bars
        self.image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        
        # Optimize scroll area for smooth dragging
        self.image_scroll.horizontalScrollBar().setSingleStep(1)
        self.image_scroll.verticalScrollBar().setSingleStep(1)
        self.image_scroll.horizontalScrollBar().setPageStep(20)
        self.image_scroll.verticalScrollBar().setPageStep(20)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(750, 500)  # 3:2 aspect ratio to better fit photos
        self.image_label.setStyleSheet("""
            background-color: transparent;
            color: #888888;
            font-size: 16px;
            font-weight: 500;
        """)
        self.image_label.setText("")  # Empty grid at startup
        
        # Enable mouse click events for zoom functionality
        self.image_label.mousePressEvent = self.image_mouse_press
        self.image_label.mouseMoveEvent = self.image_mouse_move
        self.image_label.mouseReleaseEvent = self.image_mouse_release
        self.image_label.setMouseTracking(True)
        
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(True)  # Default: allow widget to resize
        layout.addWidget(self.image_scroll)
        
        # Image name label
        self.image_name_label = QLabel("No image loaded")
        self.image_name_label.setAlignment(Qt.AlignCenter)
        self.image_name_label.setStyleSheet("""
            background-color: transparent;
            color: #aaaaaa;
            font-size: 12px;
            font-weight: 400;
            padding: 8px;
            border: none;
        """)
        layout.addWidget(self.image_name_label)
        
        # Image metadata label
        self.image_metadata_label = QLabel("")
        self.image_metadata_label.setAlignment(Qt.AlignCenter)
        self.image_metadata_label.setStyleSheet("""
            background-color: transparent;
            color: #888888;
            font-size: 10px;
            font-weight: 400;
            padding: 4px;
            border: none;
        """)
        layout.addWidget(self.image_metadata_label)
        
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
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 8, 12, 8)
        
        # Remove the load button from here as it's now in top toolbar
        
        # Add spacer
        toolbar_layout.addStretch()
        
        # Comparison buttons
        comparison_frame = QFrame()
        comparison_layout = QHBoxLayout(comparison_frame)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.setSpacing(4)
        
        # Toggle comparison button
        self.toggle_btn = QPushButton("Edited")
        self.toggle_btn.setObjectName("comparisonButton")
        self.toggle_btn.setStyleSheet("""
            #comparisonButton {
                background-color: #4a4a4a;
                border: 1px solid #666666;
                color: white;
                font-size: 11px;
                font-weight: 500;
                padding: 8px 16px;
                border-radius: 6px;
            }
            #comparisonButton:hover {
                background-color: #5a5a5a;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_comparison)
        self.toggle_btn.setEnabled(False)
        comparison_layout.addWidget(self.toggle_btn)
        
        toolbar_layout.addWidget(comparison_frame)
        layout.addWidget(toolbar)
        
        return panel
    
    def update_image_metadata(self, file_path):
        """Update image name and metadata display on same line."""
        try:
            filename = os.path.basename(file_path)
            if self.original_image is not None:
                # Get file size in MB
                file_size_bytes = os.path.getsize(file_path)
                file_size_mb = file_size_bytes / (1024 * 1024)
                
                # Get image dimensions
                height, width = self.original_image.shape[:2]
                
                # Calculate aspect ratio
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return a
                
                ratio_gcd = gcd(width, height)
                ratio_w = width // ratio_gcd
                ratio_h = height // ratio_gcd
                
                # Format combined string with name and metadata
                display_text = f"{filename}  •  {width} × {height} pixels  •  {ratio_w}:{ratio_h}  •  {file_size_mb:.1f} MB"
                self.image_name_label.setText(display_text)
            else:
                self.image_name_label.setText(filename)
        except Exception as e:
            print(f"Error updating metadata: {e}")
            self.image_name_label.setText(os.path.basename(file_path))
    
    def update_import_path_label(self):
        """Update the import path label display."""
        path = getattr(dynamic_config, 'import_folder_path', '')
        if path:
            # Show full path without prefix
            self.import_path_label.setText(path)
        else:
            self.import_path_label.setText("")
    
    def update_save_path_label(self):
        """Update the save path label display."""
        path = getattr(dynamic_config, 'export_folder_path', '')
        if path:
            # Show full path without prefix
            self.save_path_label.setText(path)
        else:
            self.save_path_label.setText("")
    
    def create_controls_panel(self):
        """Create the modern controls panel with filter buttons."""
        panel = QFrame()
        panel.setObjectName("controlsPanel")
        panel.setStyleSheet("""
            #controlsPanel {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Exposure Control Section
        exposure_frame = QFrame()
        exposure_frame.setStyleSheet("border: none; background: transparent;")
        exposure_layout = QHBoxLayout(exposure_frame)
        exposure_layout.setContentsMargins(0, 16, 0, 0)
        exposure_layout.setSpacing(12)
        
        # Exposure label on the left
        exposure_label = QLabel("Exposure")
        exposure_label.setStyleSheet("""
            font-size: 12px;
            font-weight: normal;
            color: #ffffff;
            background: transparent;
            border: none;
        """)
        exposure_label.setFixedWidth(70)  # Fixed width for alignment
        exposure_layout.addWidget(exposure_label)
        
        # Exposure slider in the middle
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setMinimum(-100)
        self.exposure_slider.setMaximum(100)
        self.exposure_slider.setValue(0)  # Start in middle
        
        # Enhanced smooth slider styling
        self.exposure_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 4px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #2a2a2a, stop:0.5 #3a3a3a, stop:1 #2a2a2a);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #888888, stop:0.5 #666666, stop:1 #444444);
                border: 2px solid #999999;
                width: 20px;
                height: 24px;
                margin: -10px 0;
                border-radius: 12px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #aaaaaa, stop:0.5 #888888, stop:1 #666666);
                border-color: #bbbbbb;
            }
            QSlider::handle:horizontal:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #0078d4, stop:0.5 #106ebe, stop:1 #005a9e);
                border-color: #0078d4;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #0078d4, stop:1 #106ebe);
                border-radius: 3px;
            }
        """)
        
        # Set up smooth value tracking
        self.exposure_slider.setTracking(True)  # Enable real-time tracking
        self.exposure_slider.valueChanged.connect(self.on_exposure_changed_smooth)
        self.exposure_slider.mouseDoubleClickEvent = self.reset_exposure_on_double_click
        self.exposure_slider.setEnabled(False)  # Disabled until image is loaded
        
        # Initialize exposure update timer for smooth updates
        self.exposure_update_timer = QTimer()
        self.exposure_update_timer.setSingleShot(True)
        self.exposure_update_timer.timeout.connect(self.apply_exposure_update)
        self.exposure_update_delay = 50  # 50ms delay for smooth updates
        exposure_layout.addWidget(self.exposure_slider)
        
        # Exposure value label on the right
        self.exposure_value_label = QLabel("0")
        self.exposure_value_label.setAlignment(Qt.AlignCenter)
        self.exposure_value_label.setStyleSheet("""
            font-size: 11px;
            color: #aaaaaa;
            background: transparent;
            border: none;
        """)
        self.exposure_value_label.setFixedWidth(30)  # Fixed width for alignment
        exposure_layout.addWidget(self.exposure_value_label)
        
        layout.addWidget(exposure_frame)
        
        # Shadow Control Section
        shadow_frame = QFrame()
        shadow_frame.setStyleSheet("border: none; background: transparent;")
        shadow_layout = QHBoxLayout(shadow_frame)
        shadow_layout.setContentsMargins(0, 16, 0, 0)
        shadow_layout.setSpacing(12)
        
        # Shadow label on the left
        shadow_label = QLabel("Shadows")
        shadow_label.setStyleSheet("""
            font-size: 12px;
            font-weight: normal;
            color: #ffffff;
            background: transparent;
            border: none;
        """)
        shadow_label.setFixedWidth(70)  # Fixed width for alignment
        shadow_layout.addWidget(shadow_label)
        
        # Shadow slider in the middle
        self.shadow_slider = QSlider(Qt.Horizontal)
        self.shadow_slider.setMinimum(-100)
        self.shadow_slider.setMaximum(100)
        self.shadow_slider.setValue(0)  # Start in middle
        
        # Enhanced smooth slider styling (same as exposure)
        self.shadow_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 4px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #2a2a2a, stop:0.5 #3a3a3a, stop:1 #2a2a2a);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #888888, stop:0.5 #666666, stop:1 #444444);
                border: 2px solid #999999;
                width: 20px;
                height: 24px;
                margin: -10px 0;
                border-radius: 12px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #aaaaaa, stop:0.5 #888888, stop:1 #666666);
                border-color: #bbbbbb;
            }
            QSlider::handle:horizontal:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #0078d4, stop:0.5 #106ebe, stop:1 #005a9e);
                border-color: #0078d4;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #0078d4, stop:1 #106ebe);
                border-radius: 3px;
            }
        """)
        
        # Set up smooth value tracking
        self.shadow_slider.setTracking(True)  # Enable real-time tracking
        self.shadow_slider.valueChanged.connect(self.on_shadow_changed_smooth)
        self.shadow_slider.mouseDoubleClickEvent = self.reset_shadow_on_double_click
        self.shadow_slider.setEnabled(False)  # Disabled until image is loaded
        
        # Initialize shadow update timer for smooth updates
        self.shadow_update_timer = QTimer()
        self.shadow_update_timer.setSingleShot(True)
        self.shadow_update_timer.timeout.connect(self.apply_shadow_update)
        self.shadow_update_delay = 50  # 50ms delay for smooth updates
        shadow_layout.addWidget(self.shadow_slider)
        
        # Shadow value label on the right
        self.shadow_value_label = QLabel("0")
        self.shadow_value_label.setAlignment(Qt.AlignCenter)
        self.shadow_value_label.setStyleSheet("""
            font-size: 11px;
            color: #aaaaaa;
            background: transparent;
            border: none;
        """)
        self.shadow_value_label.setFixedWidth(30)  # Fixed width for alignment
        shadow_layout.addWidget(self.shadow_value_label)
        
        layout.addWidget(shadow_frame)
        
        # Collapsible Filter Section
        filter_group = QGroupBox("Filters")
        filter_group.setCheckable(True)
        filter_group.setChecked(True)  # Start expanded
        filter_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: 600;
                color: #ffffff;
                background: transparent;
                border: 1px solid #404040;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background: #2d2d2d;
            }
            QGroupBox::indicator {
                width: 12px;
                height: 12px;
                margin-right: 6px;
            }
            QGroupBox::indicator:checked {
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTMgNkw2IDlMOSAzIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
            }
            QGroupBox::indicator:unchecked {
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQgNEw4IDgiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KPHBhdGggZD0iTTggNEw0IDgiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KPC9zdmc+Cgo=);
            }
        """)
        
        filter_group_layout = QVBoxLayout(filter_group)
        filter_group_layout.setContentsMargins(8, 8, 8, 8)
        
        # Filter buttons scroll area
        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # Enhanced smooth scrolling properties
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setSizeAdjustPolicy(QScrollArea.AdjustToContents)
        scroll_area.setWidgetResizable(True)
        
        # Enable kinetic scrolling for smoother experience
        scroll_area.setAttribute(Qt.WA_AcceptTouchEvents, True)
        
        # Set scroll step to smaller value for smoother scrolling
        scroll_area.verticalScrollBar().setSingleStep(10)
        scroll_area.horizontalScrollBar().setSingleStep(10)
        
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background-color: transparent;")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)  # Zero spacing between filter buttons
        
        if not self.available_filters:
            no_filters_label = QLabel("No filters found")
            no_filters_label.setAlignment(Qt.AlignCenter)
            no_filters_label.setStyleSheet("color: #888888; font-size: 14px; padding: 40px;")
            scroll_layout.addWidget(no_filters_label)
        else:
            for filter_name, filter_path in self.available_filters.items():
                btn = FilterButton(filter_name, filter_path)
                btn.update_style()  # Set initial styling
                btn.clicked.connect(lambda checked, name=filter_name, path=filter_path, button=btn: 
                                  self.toggle_filter(name, path, button))
                btn.hoverEntered.connect(self.on_filter_hover)
                btn.hoverLeft.connect(self.on_filter_hover_left)
                btn.setEnabled(False)
                scroll_layout.addWidget(btn)
                
                # Store reference for enabling/disabling
                if not hasattr(self, 'filter_buttons'):
                    self.filter_buttons = []
                self.filter_buttons.append(btn)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        filter_group_layout.addWidget(scroll_area)
        
        # Connect mouse wheel events for smoother scrolling
        scroll_area.wheelEvent = lambda event: self.smooth_wheel_event(scroll_area, event)
        
        layout.addWidget(filter_group)
        
        return panel
    
    def reset_shadow_on_double_click(self, event):
        """Reset shadow slider to 0 on double click."""
        if event.button() == Qt.LeftButton:
            self.shadow_slider.setValue(0)
    
    def reset_exposure_on_double_click(self, event):
        """Reset exposure slider to 0 on double click."""
        if event.button() == Qt.LeftButton:
            self.exposure_slider.setValue(0)
    
    def apply_shadow_to_image(self, image, shadow_value):
        """Apply shadow adjustment to an image and return the adjusted image."""
        if shadow_value == 0:
            return image
        
        try:
            # Convert to float for processing
            img_float = image.astype(np.float32) / 255.0
            
            # Calculate shadow mask (darker areas)
            # Use luminance to identify shadows
            if len(img_float.shape) == 3:
                luminance = 0.299 * img_float[:,:,2] + 0.587 * img_float[:,:,1] + 0.114 * img_float[:,:,0]
            else:
                luminance = img_float
            
            # Create shadow mask: shadows are areas with low luminance
            shadow_mask = 1.0 - luminance  # Invert so dark areas = 1, bright areas = 0
            shadow_mask = np.clip(shadow_mask ** 2, 0, 1)  # Square for better falloff
            
            # Convert shadow value (-100 to +100) to adjustment factor
            factor = shadow_value / 100.0
            
            # Apply shadow adjustment
            if len(img_float.shape) == 3:
                for channel in range(3):
                    adjustment = factor * shadow_mask
                    img_float[:,:,channel] = np.clip(img_float[:,:,channel] + adjustment, 0, 1)
            else:
                adjustment = factor * shadow_mask
                img_float = np.clip(img_float + adjustment, 0, 1)
            
            # Convert back to uint8
            adjusted_image = np.clip(img_float * 255, 0, 255).astype(np.uint8)
            
            return adjusted_image
        except Exception as e:
            print(f"Error applying shadow adjustment: {e}")
            return image
    
    def apply_exposure_to_image(self, image, exposure_value):
        """Apply exposure adjustment to an image and return the adjusted image."""
        if exposure_value == 0:
            return image
        
        try:
            # Convert exposure value (-100 to +100) to brightness factor
            factor = 1.0 + (exposure_value / 100.0)
            
            # Apply exposure adjustment to all color channels
            adjusted_image = image.astype(np.float32) * factor
            
            # Clip values to valid range [0, 255] and convert back to uint8
            adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
            
            return adjusted_image
        except Exception as e:
            print(f"Error applying exposure: {e}")
            return image
    
    def on_shadow_changed_smooth(self, value):
        """Handle shadow slider changes with smooth debounced updates."""
        self.shadow_adjustment = value
        self.shadow_value_label.setText(str(value))
        
        # Update toggle button text when adjustments change
        self.update_toggle_button_text()
        
        # Use timer to debounce rapid changes for smoother performance
        if self.original_image is not None:
            self.shadow_update_timer.stop()  # Cancel any pending update
            self.shadow_update_timer.start(self.shadow_update_delay)
            
            # If we're currently showing original mode, switch to edited to show changes
            if self.current_display_mode == "original" and value != 0:
                self.current_display_mode = "filtered"
                self.update_toggle_button_text()
    
    def apply_shadow_update(self):
        """Apply the shadow update after the debounce delay."""
        if self.original_image is not None:
            self.update_display_with_adjustments()
    
    def on_exposure_changed_smooth(self, value):
        """Handle exposure slider changes with smooth debounced updates."""
        self.exposure_adjustment = value
        self.exposure_value_label.setText(str(value))
        
        # Update toggle button text when adjustments change
        self.update_toggle_button_text()
        
        # Use timer to debounce rapid changes for smoother performance
        if self.original_image is not None:
            self.exposure_update_timer.stop()  # Cancel any pending update
            self.exposure_update_timer.start(self.exposure_update_delay)
            
            # If we're currently showing original mode, switch to edited to show changes
            if self.current_display_mode == "original" and value != 0:
                self.current_display_mode = "filtered"
                self.update_toggle_button_text()
    
    def apply_exposure_update(self):
        """Apply the exposure update after the debounce delay."""
        if self.original_image is not None:
            self.update_display_with_adjustments()
    
    def on_exposure_changed(self, value):
        """Handle exposure slider changes."""
        self.exposure_adjustment = value
        self.exposure_value_label.setText(str(value))
        
        # Apply exposure adjustment to current image
        if self.original_image is not None:
            self.update_display_with_adjustments()
    
    def update_display_with_adjustments(self):
        """Update the displayed image with current exposure and shadow adjustments."""
        try:
            if self.current_display_mode == "original":
                # Show pure original image without any adjustments
                self.display_image(self.original_image)
            else:
                # Show image with all adjustments and filters applied
                # Start with the base image (original or filtered)
                if self.current_filtered_image is not None:
                    base_image = self.current_filtered_image.copy()
                else:
                    base_image = self.original_image.copy()
                
                # Apply adjustments in order: exposure first, then shadows
                adjusted_image = base_image
                
                # Apply exposure adjustment
                if self.exposure_adjustment != 0:
                    adjusted_image = self.apply_exposure_to_image(adjusted_image, self.exposure_adjustment)
                
                # Apply shadow adjustment
                if self.shadow_adjustment != 0:
                    adjusted_image = self.apply_shadow_to_image(adjusted_image, self.shadow_adjustment)
                
                self.display_image(adjusted_image)
                
        except Exception as e:
            print(f"Error applying adjustments: {e}")
    
    def update_display_with_exposure(self):
        """Update the displayed image with current exposure adjustment."""
        try:
            # Start with the base image (original or filtered)
            if self.current_display_mode == "filtered" and self.current_filtered_image is not None:
                base_image = self.current_filtered_image.copy()
            else:
                base_image = self.original_image.copy()
            
            # Apply exposure adjustment
            if self.exposure_adjustment != 0:
                # Convert exposure value (-100 to +100) to brightness factor
                # Positive values brighten, negative values darken
                factor = 1.0 + (self.exposure_adjustment / 100.0)
                
                # Apply exposure adjustment to all color channels
                adjusted_image = base_image.astype(np.float32) * factor
                
                # Clip values to valid range [0, 255] and convert back to uint8
                adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
                
                self.display_image(adjusted_image)
            else:
                # No adjustment needed
                self.display_image(base_image)
                
        except Exception as e:
            print(f"Error applying exposure: {e}")
    
    def get_available_filters(self):
        """Get all available filter files."""
        filters = {}
        
        if not os.path.exists(self.filters_dir):
            return filters
        
        for file in os.listdir(self.filters_dir):
            if file.endswith('.py') and not file.startswith('__'):
                filter_name = file.replace('.py', '').replace('_', ' ').title()
                # Clean up some names
                filter_name = filter_name.replace('Kodak ', 'Kodak ')
                filter_name = filter_name.replace('Ilford ', 'Ilford ')
                filter_name = filter_name.replace(' Copy', '')
                
                filter_path = os.path.join(self.filters_dir, file)
                filters[filter_name] = filter_path
        
        return filters
    
    def update_dynamic_config(self, key, value):
        """Update a value in dynamic_config.py file."""
        try:
            config_file = "dynamic_config.py"
            
            print(f"Updating {key} = {value}")  # Debug output
            
            # Read current content
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"Current content: {repr(content)}")  # Debug output
            
            # Update the specific line
            lines = content.split('\n')
            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f'{key} ='):
                    lines[i] = f'{key} = "{value}"'
                    updated = True
                    print(f"Updated line {i}: {lines[i]}")  # Debug output
                    break
            
            if not updated:
                print(f"Could not find line for key: {key}")  # Debug output
                return
                
            # Write back to file
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
            # Update the imported module's attribute
            setattr(dynamic_config, key, value)
            print(f"Successfully updated {key}")  # Debug output
            
        except Exception as e:
            print(f"Error updating dynamic config: {e}")
            import traceback
            traceback.print_exc()
    
    def load_image_unicode(self, file_path):
        """Load image that handles Unicode file paths properly."""
        try:
            # Use numpy to read file as bytes, then decode with OpenCV
            # This method handles Unicode characters in file paths
            img_array = np.fromfile(file_path, dtype=np.uint8)
            if img_array.size == 0:
                return None
            
            # Decode the image from bytes
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
            
        except Exception as e:
            print(f"Error loading Unicode image: {e}")
            return None
    
    def save_image_unicode(self, file_path, image):
        """Save image that handles Unicode file paths properly."""
        try:
            # Encode image to memory buffer
            is_success, buffer = cv2.imencode('.jpg', image)
            if not is_success:
                return False
            
            # Write buffer to file using Python's file handling (supports Unicode paths)
            with open(file_path, 'wb') as f:
                f.write(buffer.tobytes())
            
            return True
            
        except Exception as e:
            print(f"Error saving Unicode image: {e}")
            return False
    
    def show_temp_popup(self, message):
        """Show a temporary popup message that disappears after 1 second."""
        popup = QMessageBox(self)
        popup.setWindowTitle("")
        popup.setText(message)
        popup.setIcon(QMessageBox.NoIcon)  # Remove any icon
        popup.setStandardButtons(QMessageBox.NoButton)  # No buttons
        popup.setStyleSheet("""
            QMessageBox {
                background-color: #2d2d2d;
                color: #ffffff;
                font-size: 16px;
                font-weight: 600;
                border: 2px solid #0078d4;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        
        # Show the popup
        popup.show()
        
        # Set up timer to close after 1 second
        QTimer.singleShot(1000, popup.accept)
    
    def load_image(self):
        """Load an image file."""
        # Use saved import path, fallback to root if empty
        default_dir = getattr(dynamic_config, 'import_folder_path', '') or '/'
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image", 
            default_dir, 
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            # Save the directory path for future use
            folder_path = os.path.dirname(file_path)
            self.update_dynamic_config('import_folder_path', folder_path)
            self.update_import_path_label()
            try:
                # Debug: Print file path and check if file exists
                print(f"Attempting to load: {file_path}")
                print(f"File exists: {os.path.exists(file_path)}")
                
                # Load image with OpenCV - handle Unicode paths
                self.original_image = self.load_image_unicode(file_path)
                
                if self.original_image is None:
                    # More detailed error information
                    if not os.path.exists(file_path):
                        QMessageBox.critical(self, "Error", f"File does not exist:\n{file_path}")
                    else:
                        # Try to determine file format
                        file_ext = os.path.splitext(file_path)[1].lower()
                        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                        
                        if file_ext not in supported_formats:
                            QMessageBox.critical(self, "Error", 
                                f"Unsupported file format: {file_ext}\n"
                                f"Supported formats: {', '.join(supported_formats)}")
                        else:
                            QMessageBox.critical(self, "Error", 
                                f"Could not load the selected image.\n"
                                f"The file might be corrupted or in an unsupported format.\n"
                                f"File: {file_path}")
                    # Reset image name on failed load
                    if hasattr(self, 'image_name_label'):
                        self.image_name_label.setText("No image loaded")
                    # Reset original filename and zoom state
                    self.original_filename = None
                    self.is_zoomed = False
                    self.original_pixmap = None
                    self.dragging = False
                    return
                
                # Store original filename for save suggestions
                self.original_filename = os.path.basename(file_path)
                
                # Reset state
                self.current_filtered_image = None
                self.current_display_mode = "original"
                self.preview_cache.clear()  # Clear preview cache for new image
                self.is_hovering = False
                self.currently_hovering_filter = None
                
                # Reset zoom state
                self.is_zoomed = False
                self.original_pixmap = None
                self.dragging = False
                
                # Reset filter selection
                if self.selected_filter_button:
                    self.selected_filter_button.set_selected(False)
                self.selected_filter = None
                self.selected_filter_button = None
                
                # Display the image
                self.display_image(self.original_image)
                
                # Enable filter buttons and save button
                if hasattr(self, 'filter_buttons'):
                    for btn in self.filter_buttons:
                        btn.setEnabled(True)
                
                # Enable save button
                if hasattr(self, 'save_btn'):
                    self.save_btn.setEnabled(True)
                
                # Enable toggle button
                self.toggle_btn.setEnabled(True)
                self.update_toggle_button_text()
                
                # Reset and enable exposure and shadow sliders
                if hasattr(self, 'exposure_slider'):
                    self.exposure_slider.setValue(0)  # Reset to 0 for new image
                    self.exposure_adjustment = 0
                    if hasattr(self, 'exposure_value_label'):
                        self.exposure_value_label.setText('0')
                    self.exposure_slider.setEnabled(True)
                
                if hasattr(self, 'shadow_slider'):
                    self.shadow_slider.setValue(0)  # Reset to 0 for new image
                    self.shadow_adjustment = 0
                    if hasattr(self, 'shadow_value_label'):
                        self.shadow_value_label.setText('0')
                    self.shadow_slider.setEnabled(True)
                
                # Update image name and metadata display
                self.update_image_metadata(file_path)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                # Reset image name on error
                if hasattr(self, 'image_name_label'):
                    self.image_name_label.setText("No image loaded")
                if hasattr(self, 'image_metadata_label'):
                    self.image_metadata_label.setText("")
                # Reset original filename and zoom state
                self.original_filename = None
                self.is_zoomed = False
                self.original_pixmap = None
                self.dragging = False
    
    def get_next_available_filename(self, folder_path, base_name, extension):
        """Find the next available filename with incrementing number suffix."""
        counter = 1
        while True:
            if counter == 1:
                # First try with _01
                new_filename = f"{base_name}_01{extension}"
            else:
                # Increment with zero-padding
                new_filename = f"{base_name}_{counter:02d}{extension}"
            
            full_path = os.path.join(folder_path, new_filename)
            if not os.path.exists(full_path):
                return new_filename
            counter += 1
            
            # Safety break to avoid infinite loop (max 999 files)
            if counter > 999:
                # Fallback to timestamp if too many files
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                return f"{base_name}_{timestamp}{extension}"
    
    def save_image(self):
        """Save the current image (filtered or original) with Save As dialog."""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded to save.")
            return
        
        # Use saved export path, fallback to root if empty
        default_dir = getattr(dynamic_config, 'export_folder_path', '') or '/'
        
        # Create suggested filename based on original image name
        if hasattr(self, 'original_filename') and self.original_filename:
            # Get base name without extension
            base_name = os.path.splitext(self.original_filename)[0]
            original_ext = os.path.splitext(self.original_filename)[1].lower()
            
            # Use original extension or default to .jpg
            if original_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                file_ext = original_ext
            else:
                file_ext = '.jpg'
            
            # Find next available filename
            suggested_filename = self.get_next_available_filename(default_dir, base_name, file_ext)
        else:
            # Fallback to timestamp-based naming
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.current_display_mode == "filtered" and self.current_filtered_image is not None:
                suffix = "_filtered"
            else:
                suffix = "_original"
            suggested_filename = f"image_{timestamp}{suffix}.jpg"
        
        # Combine default directory with suggested filename
        default_path = os.path.join(default_dir, suggested_filename)
        
        # Determine which image to save
        if self.current_display_mode == "filtered" and self.current_filtered_image is not None:
            image_to_save = self.current_filtered_image.copy()
        else:
            image_to_save = self.original_image.copy()
        
        # Apply exposure and shadow adjustments before saving
        if hasattr(self, 'exposure_adjustment') and self.exposure_adjustment != 0:
            factor = 1.0 + (self.exposure_adjustment / 100.0)
            image_to_save = image_to_save.astype(np.float32) * factor
            image_to_save = np.clip(image_to_save, 0, 255).astype(np.uint8)
        
        if hasattr(self, 'shadow_adjustment') and self.shadow_adjustment != 0:
            image_to_save = self.apply_shadow_to_image(image_to_save, self.shadow_adjustment)
        
        # Open Save As dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image As",
            default_path,
            "JPEG Images (*.jpg);;PNG Images (*.png);;All Images (*.jpg *.png *.bmp *.tiff)"
        )
        
        if file_path:
            # Save the directory path for future use
            folder_path = os.path.dirname(file_path)
            self.update_dynamic_config('export_folder_path', folder_path)
            self.update_save_path_label()
            
            try:
                # Save the image using Unicode-safe method
                success = self.save_image_unicode(file_path, image_to_save)
                
                if success:
                    pass  # Image saved successfully without popup
                else:
                    QMessageBox.critical(self, "Error", "Failed to save image.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving image: {str(e)}")
                print(f"Error saving image: {e}")
    
    def image_mouse_press(self, event):
        """Handle mouse press events on image."""
        if self.original_image is not None and self.original_pixmap is not None:
            if event.button() == Qt.LeftButton:
                # Store left-click start position
                self.left_click_start_pos = event.pos()
                
                # Calculate click position relative to the image for zoom centering
                if not self.is_zoomed:
                    self.calculate_zoom_center_point(event.pos())
                    
                if self.is_zoomed:
                    # Prepare for potential dragging when zoomed
                    self.last_pan_point = event.pos()
    
    def image_mouse_move(self, event):
        """Handle mouse move events for dragging."""
        if event.buttons() == Qt.LeftButton and self.is_zoomed:
            # Start dragging if we moved far enough from start position
            if not self.dragging:
                move_distance = (event.pos() - self.left_click_start_pos).manhattanLength()
                if move_distance > 3:  # Reduced threshold for more responsive drag start
                    self.dragging = True
                    self.image_label.setCursor(QCursor(Qt.ClosedHandCursor))
            
            if self.dragging and hasattr(self, 'last_pan_point'):
                # Calculate movement delta
                delta = event.pos() - self.last_pan_point
                
                # Get current scroll bar positions
                h_scroll = self.image_scroll.horizontalScrollBar()
                v_scroll = self.image_scroll.verticalScrollBar()
                
                # Apply movement with 1:1 ratio for natural feel
                new_h = h_scroll.value() - delta.x()
                new_v = v_scroll.value() - delta.y()
                
                # Update scroll positions immediately for smooth performance
                h_scroll.setValue(new_h)
                v_scroll.setValue(new_v)
                
                # Update immediately for smoother visual feedback
                self.image_scroll.update()
                
            # Always update last pan point for next movement
            self.last_pan_point = event.pos()
    
    def image_mouse_release(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton:
            if self.dragging:
                # End dragging
                self.dragging = False
                # Return to hand cursor when zoomed
                if self.is_zoomed:
                    self.image_label.setCursor(QCursor(Qt.OpenHandCursor))
                else:
                    self.image_label.setCursor(QCursor(Qt.ArrowCursor))
            else:
                # Short left-click without drag = toggle zoom
                move_distance = (event.pos() - self.left_click_start_pos).manhattanLength()
                if move_distance <= 5:  # Consider it a short click
                    self.toggle_zoom()
    
    def calculate_zoom_center_point(self, click_pos):
        """Calculate the point in original image coordinates that should be centered when zooming."""
        try:
            # Get current pixmap and its display size
            current_pixmap = self.image_label.pixmap()
            if not current_pixmap or self.original_pixmap is None:
                return
                
            # Get the displayed image rectangle within the label
            label_rect = self.image_label.rect()
            pixmap_rect = current_pixmap.rect()
            
            # Calculate the actual display rectangle (centered and scaled)
            scale_x = pixmap_rect.width() / self.original_pixmap.width()
            scale_y = pixmap_rect.height() / self.original_pixmap.height()
            scale = min(scale_x, scale_y)  # Keep aspect ratio
            
            scaled_width = int(self.original_pixmap.width() * scale)
            scaled_height = int(self.original_pixmap.height() * scale)
            
            # Center the scaled image in the label
            x_offset = (label_rect.width() - scaled_width) // 2
            y_offset = (label_rect.height() - scaled_height) // 2
            
            # Convert click position to image coordinates
            image_x = (click_pos.x() - x_offset) / scale
            image_y = (click_pos.y() - y_offset) / scale
            
            # Clamp to image bounds
            image_x = max(0, min(image_x, self.original_pixmap.width()))
            image_y = max(0, min(image_y, self.original_pixmap.height()))
            
            self.zoom_center_point = QPoint(int(image_x), int(image_y))
            
        except Exception as e:
            print(f"Error calculating zoom center: {e}")
            self.zoom_center_point = None
    
    def image_click_event(self, event):
        """Handle mouse click on image for zoom functionality."""
        if self.original_image is not None and self.original_pixmap is not None:
            self.toggle_zoom()
    
    def toggle_zoom(self):
        """Toggle between fit-to-window and 100% zoom."""
        if self.is_zoomed:
            # Zoom out - fit to window
            self.zoom_fit_to_window()
        else:
            # Zoom in - 100% size
            self.zoom_100_percent()
        
        self.is_zoomed = not self.is_zoomed
    
    def zoom_fit_to_window(self):
        """Zoom image to fit the display area."""
        if self.original_pixmap:
            # Enable widget resizing for fit-to-window mode
            self.image_scroll.setWidgetResizable(True)
            
            label_size = self.image_label.size()
            scaled_pixmap = self.original_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            # Set normal cursor when not zoomed
            self.image_label.setCursor(QCursor(Qt.ArrowCursor))
    
    def zoom_100_percent(self):
        """Display image at 100% (original) size."""
        if self.original_pixmap:
            # Disable widget resizing for 100% zoom to enable scrolling
            self.image_scroll.setWidgetResizable(False)
            
            # Set label size to image size for proper scrolling
            self.image_label.resize(self.original_pixmap.size())
            self.image_label.setPixmap(self.original_pixmap)
            
            # Center the view on the clicked point if available
            if self.zoom_center_point:
                QTimer.singleShot(10, self.center_view_on_point)  # Delay to ensure scroll area is updated
            
            # Set hand cursor when zoomed
            self.image_label.setCursor(QCursor(Qt.OpenHandCursor))
    
    def center_view_on_point(self):
        """Center the scroll view on the previously clicked point."""
        if self.zoom_center_point:
            # Get scroll area dimensions
            scroll_area_size = self.image_scroll.viewport().size()
            
            # Calculate scroll position to center the clicked point
            center_x = self.zoom_center_point.x() - scroll_area_size.width() // 2
            center_y = self.zoom_center_point.y() - scroll_area_size.height() // 2
            
            # Set scroll bar positions
            h_scroll = self.image_scroll.horizontalScrollBar()
            v_scroll = self.image_scroll.verticalScrollBar()
            
            h_scroll.setValue(center_x)
            v_scroll.setValue(center_y)
    
    def display_image(self, cv_image):
        """Display a CV2 image in the QLabel."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and store original
        pixmap = QPixmap.fromImage(qt_image)
        self.original_pixmap = pixmap  # Store for zoom functionality
        
        # Reset zoom state when displaying new image
        self.is_zoomed = False
        
        # Ensure widget is resizable for fit-to-window mode
        self.image_scroll.setWidgetResizable(True)
        
        # Scale pixmap to fit as large as possible while keeping native aspect ratio
        label_size = self.image_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def apply_filter(self, filter_name, filter_path):
        """Apply a filter to the current image."""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        
        if self.filter_worker and self.filter_worker.isRunning():
            return  # Don't start new filter while one is running
        
        # Update status
        
        # Disable filter buttons temporarily
        if hasattr(self, 'filter_buttons'):
            for btn in self.filter_buttons:
                btn.setEnabled(False)
        
        # Apply current exposure and shadow adjustments to the original image before filtering
        base_image = self.original_image.copy()
        if hasattr(self, 'exposure_adjustment') and self.exposure_adjustment != 0:
            base_image = self.apply_exposure_to_image(base_image, self.exposure_adjustment)
        if hasattr(self, 'shadow_adjustment') and self.shadow_adjustment != 0:
            base_image = self.apply_shadow_to_image(base_image, self.shadow_adjustment)
        
        # Start filter worker thread
        self.filter_worker = FilterWorker(base_image, filter_path, filter_name)
        self.filter_worker.filterApplied.connect(self.on_filter_applied)
        self.filter_worker.start()
    
    def toggle_filter(self, filter_name, filter_path, button):
        """Toggle filter selection - apply or remove filter."""
        if self.original_image is None:
            return
        
        # Check if this filter is already selected
        if self.selected_filter == filter_name:
            # Deselect the filter - return to original
            self.deselect_filter()
        else:
            # Select this filter and deselect any previous one
            self.select_filter(filter_name, filter_path, button)
    
    def select_filter(self, filter_name, filter_path, button):
        """Select a filter and apply it."""
        # Deselect previous filter if any
        if self.selected_filter_button:
            self.selected_filter_button.set_selected(False)
        
        # Select new filter
        self.selected_filter = filter_name
        self.selected_filter_button = button
        button.set_selected(True)
        
        # Apply the filter
        self.apply_filter(filter_name, filter_path)
    
    def deselect_filter(self):
        """Deselect current filter and return to original image."""
        if self.selected_filter_button:
            self.selected_filter_button.set_selected(False)
        
        self.selected_filter = None
        self.selected_filter_button = None
        self.current_filtered_image = None
        self.current_display_mode = "original"
        
        # Update display with any remaining adjustments (exposure/shadow)
        if self.original_image is not None:
            self.update_display_with_adjustments()
        
        # Update toggle button based on remaining adjustments
        self.update_toggle_button_text()
    
    def on_filter_applied(self, filtered_image, filter_name):
        """Handle filter application completion."""
        self.current_filtered_image = filtered_image
        self.current_display_mode = "filtered"
        
        # Display the filtered image (exposure is already applied)
        self.display_image(filtered_image)
        
        # Enable toggle button and update text
        if hasattr(self, 'toggle_btn'):
            self.toggle_btn.setEnabled(True)
            self.update_toggle_button_text()
        
        # Re-enable filter buttons
        if hasattr(self, 'filter_buttons'):
            for btn in self.filter_buttons:
                btn.setEnabled(True)
        
        # Update status
    
    def on_filter_hover(self, filter_name, filter_path):
        """Handle filter button hover - show preview."""
        if self.original_image is None or (self.filter_worker and self.filter_worker.isRunning()):
            return
            
        # Disable hover preview if a filter is already selected
        if self.selected_filter is not None:
            return
            
        self.is_hovering = True
        self.currently_hovering_filter = filter_name
        
        # Create cache key that includes exposure and shadow values
        exposure_val = getattr(self, 'exposure_adjustment', 0)
        shadow_val = getattr(self, 'shadow_adjustment', 0)
        cache_key = f"{filter_name}_exp_{exposure_val}_shd_{shadow_val}"
        
        # Check if preview is cached with current exposure
        if cache_key in self.preview_cache:
            self.display_image(self.preview_cache[cache_key])
        else:
            # Generate preview in background
            self.generate_preview_with_cache_key(filter_name, filter_path, cache_key)
    
    def on_filter_hover_left(self):
        """Handle filter button hover leave - restore current view."""
        self.is_hovering = False
        self.currently_hovering_filter = None
        
        # Very short delay to allow smooth transition to next filter
        QTimer.singleShot(25, self._restore_current_view)
    
    def _restore_current_view(self):
        """Restore the current view if not hovering any filter."""
        # Only restore if we're truly not hovering (not just between filters)
        if not self.is_hovering and self.currently_hovering_filter is None:
            if self.current_display_mode == "filtered" and self.current_filtered_image is not None:
                self.display_image(self.current_filtered_image)
            elif self.original_image is not None:
                self.display_image(self.original_image)
    
    def generate_preview_with_cache_key(self, filter_name, filter_path, cache_key):
        """Generate a preview of the filter with a specific cache key."""
        if self.filter_worker and self.filter_worker.isRunning():
            return
        
        # Create a smaller version for faster preview generation
        base_image = self.get_preview_size_image(self.original_image)
        
        # Apply current exposure and shadow adjustments to the base image before filtering
        if hasattr(self, 'exposure_adjustment') and self.exposure_adjustment != 0:
            base_image = self.apply_exposure_to_image(base_image, self.exposure_adjustment)
        if hasattr(self, 'shadow_adjustment') and self.shadow_adjustment != 0:
            base_image = self.apply_shadow_to_image(base_image, self.shadow_adjustment)
        
        # Start preview worker with cache key
        self.filter_worker = FilterWorkerWithCacheKey(base_image, filter_path, filter_name, cache_key)
        self.filter_worker.filterApplied.connect(self.on_preview_generated_with_key)
        self.filter_worker.start()
    
    def generate_preview(self, filter_name, filter_path):
        """Generate a preview of the filter and cache it."""
        if self.filter_worker and self.filter_worker.isRunning():
            return
        
        # Create a smaller version for faster preview generation
        base_image = self.get_preview_size_image(self.original_image)
        
        # Apply current exposure and shadow adjustments to the base image before filtering
        if hasattr(self, 'exposure_adjustment') and self.exposure_adjustment != 0:
            base_image = self.apply_exposure_to_image(base_image, self.exposure_adjustment)
        if hasattr(self, 'shadow_adjustment') and self.shadow_adjustment != 0:
            base_image = self.apply_shadow_to_image(base_image, self.shadow_adjustment)
        
        # Start preview worker
        self.filter_worker = FilterWorker(base_image, filter_path, filter_name)
        self.filter_worker.filterApplied.connect(self.on_preview_generated)
        self.filter_worker.start()
    
    def get_preview_size_image(self, image, max_size=800):
        """Resize image for faster preview generation."""
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image
        
        # Calculate new dimensions
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def on_preview_generated(self, filtered_image, filter_name):
        """Handle preview generation completion."""
        # Cache the preview
        self.preview_cache[filter_name] = filtered_image
        
        # Display if still hovering over this filter
        if self.is_hovering:
            self.display_image(filtered_image)
    
    def on_preview_generated_with_key(self, filtered_image, filter_name, cache_key):
        """Handle preview generation completion with cache key."""
        # Cache the preview with the specific cache key
        self.preview_cache[cache_key] = filtered_image
        
        # Display if still hovering over this filter
        if self.is_hovering:
            self.display_image(filtered_image)
    
    def show_before(self):
        """Show the original image."""
        if self.original_image is not None:
            self.display_image(self.original_image)
            self.current_display_mode = "original"
    
    def show_after(self):
        """Show the filtered image."""
        if self.current_filtered_image is not None:
            self.display_image(self.current_filtered_image)
            self.current_display_mode = "filtered"
        else:
            # No filter applied yet, show original
            self.show_before()
    
    def toggle_comparison(self):
        """Toggle between original and edited (filtered + adjustments) image."""
        if self.original_image is None:
            return
            
        if self.current_display_mode == "original":
            # Currently showing original, switch to edited
            if self.current_filtered_image is not None or self.exposure_adjustment != 0 or self.shadow_adjustment != 0:
                self.current_display_mode = "filtered"
                self.update_display_with_adjustments()
            else:
                # No edits to show, stay on original
                return
        else:
            # Currently showing edited, switch to pure original
            self.current_display_mode = "original"
            self.update_display_with_adjustments()
            
        # Update button text
        self.update_toggle_button_text()
    
    def update_toggle_button_text(self):
        """Update the toggle button text based on current display mode and available edits."""
        if hasattr(self, 'toggle_btn'):
            # Check if any edits exist (filter or adjustments)
            has_edits = (self.current_filtered_image is not None or 
                        self.exposure_adjustment != 0 or 
                        self.shadow_adjustment != 0)
            
            if not has_edits:
                # No edits available, disable toggle
                self.toggle_btn.setText("Original")
                self.toggle_btn.setEnabled(False)
            else:
                # Edits available, enable toggle with appropriate text
                self.toggle_btn.setEnabled(True)
                if self.current_display_mode == "original":
                    self.toggle_btn.setText("Original")
                else:
                    self.toggle_btn.setText("Edited")
    
    def smooth_wheel_event(self, scroll_area, event):
        """Handle mouse wheel events for smoother scrolling with animation."""
        # Get the scroll bar
        scroll_bar = scroll_area.verticalScrollBar()
        
        # Calculate smooth scroll amount based on wheel delta
        # Standard wheel delta is 120 units per "notch"
        delta = event.angleDelta().y()
        
        # Apply smooth scrolling with smaller steps for better control
        smooth_scroll_amount = int(delta / 6)  # Even more granular control
        
        # Get current scroll position
        current_value = scroll_bar.value()
        target_value = current_value - smooth_scroll_amount
        
        # Ensure we stay within bounds
        target_value = max(scroll_bar.minimum(), min(scroll_bar.maximum(), target_value))
        
        # Apply smooth animated scrolling
        self.animate_scroll(scroll_bar, current_value, target_value)
        
        # Accept the event to prevent default handling
        event.accept()
    
    def animate_scroll(self, scroll_bar, start_value, end_value, duration=150):
        """Animate scroll bar movement for smoother scrolling."""
        if not hasattr(self, 'scroll_animation_timer'):
            self.scroll_animation_timer = QTimer()
            self.scroll_animation_timer.timeout.connect(self.update_scroll_animation)
        
        # Stop any existing animation
        if self.scroll_animation_timer.isActive():
            self.scroll_animation_timer.stop()
        
        # Set up animation parameters
        self.scroll_animation_start = start_value
        self.scroll_animation_end = end_value
        self.scroll_animation_current = start_value
        self.scroll_animation_scroll_bar = scroll_bar
        self.scroll_animation_step_size = (end_value - start_value) / (duration / 16)  # 60 FPS
        self.scroll_animation_steps_remaining = duration // 16
        
        # Start animation
        self.scroll_animation_timer.start(16)  # ~60 FPS
    
    def update_scroll_animation(self):
        """Update scroll animation frame."""
        if self.scroll_animation_steps_remaining <= 0:
            # Animation complete
            self.scroll_animation_timer.stop()
            self.scroll_animation_scroll_bar.setValue(int(self.scroll_animation_end))
            return
        
        # Calculate eased position (ease-out curve for natural feel)
        progress = 1.0 - (self.scroll_animation_steps_remaining / (150 / 16))
        eased_progress = 1 - (1 - progress) ** 3  # Ease-out cubic
        
        current_pos = self.scroll_animation_start + (self.scroll_animation_end - self.scroll_animation_start) * eased_progress
        self.scroll_animation_scroll_bar.setValue(int(current_pos))
        
        self.scroll_animation_steps_remaining -= 1


def main():
    """Main function to run the GUI application."""
    try:
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("Film Filter Studio")
        app.setOrganizationName("Film Filters")
        
        # Create and show the main window
        window = FilmFilterGUI()
        window.show()
        
        # Start the event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()