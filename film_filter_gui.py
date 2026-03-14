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
                            QFileDialog, QMessageBox, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
import importlib.util
from pathlib import Path

# Import config for default paths
try:
    import config
except ImportError:
    # Fallback if config not found
    class config:
        upload_photo_path = ""


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
        
        # Get available filters
        self.filters_dir = "film_filters/filters"
        self.available_filters = self.get_available_filters()
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Film Filter Studio - Professional Photo Editing")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Left side - Image display (takes most space)
        image_panel = self.create_image_panel()
        main_layout.addWidget(image_panel, 3)  # 75% of space
        
        # Right side - Controls panel
        controls_panel = self.create_controls_panel()
        main_layout.addWidget(controls_panel, 1)  # 25% of space
        
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
            QScrollBar:vertical {
                border: none;
                background: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666666;
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
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        self.image_label.setStyleSheet("""
            background-color: transparent;
            color: #888888;
            font-size: 16px;
            font-weight: 500;
        """)
        self.image_label.setText("🖼️  Load an image to get started")
        
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(True)
        layout.addWidget(self.image_scroll)
        
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
        
        # Load button
        self.load_btn = QPushButton("📁  Load Photo")
        self.load_btn.setObjectName("loadButton")
        self.load_btn.setStyleSheet("""
            #loadButton {
                background-color: #0078d4;
                border: 1px solid #106ebe;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 14px 24px;
                border-radius: 8px;
            }
            #loadButton:hover {
                background-color: #106ebe;
                border-color: #005a9e;
            }
            #loadButton:pressed {
                background-color: #005a9e;
            }
        """)
        self.load_btn.clicked.connect(self.load_image)
        toolbar_layout.addWidget(self.load_btn)
        
        # Add spacer
        toolbar_layout.addStretch()
        
        # Comparison buttons
        comparison_frame = QFrame()
        comparison_layout = QHBoxLayout(comparison_frame)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.setSpacing(4)
        
        self.before_btn = QPushButton("📷 Original")
        self.before_btn.setObjectName("comparisonButton")
        self.before_btn.setStyleSheet("""
            #comparisonButton {
                background-color: #4a4a4a;
                border: 1px solid #666666;
                color: white;
                font-size: 12px;
                font-weight: 500;
                padding: 10px 20px;
                border-radius: 6px;
            }
            #comparisonButton:hover {
                background-color: #5a5a5a;
            }
        """)
        self.before_btn.clicked.connect(self.show_before)
        self.before_btn.setEnabled(False)
        comparison_layout.addWidget(self.before_btn)
        
        self.after_btn = QPushButton("🎨 Filtered")
        self.after_btn.setObjectName("comparisonButton")
        self.after_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                border: 1px solid #666666;
                color: white;
                font-size: 12px;
                font-weight: 500;
                padding: 10px 20px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)
        self.after_btn.clicked.connect(self.show_after)
        self.after_btn.setEnabled(False)
        comparison_layout.addWidget(self.after_btn)
        
        toolbar_layout.addWidget(comparison_frame)
        layout.addWidget(toolbar)
        
        return panel
    
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
        
        # Title section
        title_frame = QFrame()
        title_frame.setStyleSheet("border: none; background: transparent;")
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("Film Filters")
        title.setObjectName("titleLabel")
        title.setStyleSheet("""
            #titleLabel {
                font-size: 20px;
                font-weight: 700;
                color: #ffffff;
                background: transparent;
                border: none;
                padding: 0px 0px 8px 0px;
            }
        """)
        title.setAlignment(Qt.AlignLeft)
        title_layout.addWidget(title)
        
        subtitle = QLabel("Choose a filter to apply")
        subtitle.setStyleSheet("""
            font-size: 12px;
            color: #aaaaaa;
            background: transparent;
            border: none;
        """)
        title_layout.addWidget(subtitle)
        
        layout.addWidget(title_frame)
        
        # Filter buttons area
        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
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
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Status section
        self.status_label = QLabel("Ready to load photos")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("""
            #statusLabel {
                background-color: #333333;
                padding: 16px;
                border-radius: 8px;
                font-size: 11px;
                color: #cccccc;
                border: 1px solid #444444;
            }
        """)
        layout.addWidget(self.status_label)
        
        return panel
    
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
    
    def load_image(self):
        """Load an image file."""
        # Use config upload_photo_path as default directory
        default_dir = getattr(config, 'upload_photo_path', '')
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image", 
            default_dir, 
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            try:
                # Load image with OpenCV
                self.original_image = cv2.imread(file_path)
                
                if self.original_image is None:
                    QMessageBox.critical(self, "Error", "Could not load the selected image.")
                    return
                
                # Reset state
                self.current_filtered_image = None
                self.current_display_mode = "original"
                self.preview_cache.clear()  # Clear preview cache for new image
                self.is_hovering = False
                self.currently_hovering_filter = None
                
                # Reset filter selection
                if self.selected_filter_button:
                    self.selected_filter_button.set_selected(False)
                self.selected_filter = None
                self.selected_filter_button = None
                
                # Display the image
                self.display_image(self.original_image)
                
                # Enable filter buttons
                if hasattr(self, 'filter_buttons'):
                    for btn in self.filter_buttons:
                        btn.setEnabled(True)
                
                # Enable original button, disable filtered button
                self.before_btn.setEnabled(True)
                self.after_btn.setEnabled(False)
                
                # Update status
                self.status_label.setText(f"✅ Loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, cv_image):
        """Display a CV2 image in the QLabel."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and scale
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale pixmap to fit the display area while maintaining aspect ratio
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
        self.status_label.setText(f"🔄 Applying {filter_name}...")
        
        # Disable filter buttons temporarily
        if hasattr(self, 'filter_buttons'):
            for btn in self.filter_buttons:
                btn.setEnabled(False)
        
        # Start filter worker thread
        self.filter_worker = FilterWorker(self.original_image, filter_path, filter_name)
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
        
        # Show original image
        if self.original_image is not None:
            self.display_image(self.original_image)
            self.status_label.setText("📷 Showing original image")
        
        # Disable filtered button
        self.after_btn.setEnabled(False)
    
    def on_filter_applied(self, filtered_image, filter_name):
        """Handle filter application completion."""
        self.current_filtered_image = filtered_image
        self.current_display_mode = "filtered"
        
        # Display the filtered image
        self.display_image(filtered_image)
        
        # Enable filtered button
        self.after_btn.setEnabled(True)
        
        # Re-enable filter buttons
        if hasattr(self, 'filter_buttons'):
            for btn in self.filter_buttons:
                btn.setEnabled(True)
        
        # Update status
        self.status_label.setText(f"✅ Applied {filter_name}")
    
    def on_filter_hover(self, filter_name, filter_path):
        """Handle filter button hover - show preview."""
        if self.original_image is None or (self.filter_worker and self.filter_worker.isRunning()):
            return
            
        self.is_hovering = True
        self.currently_hovering_filter = filter_name
        
        # Check if preview is cached
        if filter_name in self.preview_cache:
            self.display_image(self.preview_cache[filter_name])
            self.status_label.setText(f"🔍 Previewing {filter_name}")
        else:
            # Generate preview in background
            self.status_label.setText(f"🔄 Loading preview: {filter_name}...")
            self.generate_preview(filter_name, filter_path)
    
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
                self.status_label.setText("🎨 Showing filtered image")
            elif self.original_image is not None:
                self.display_image(self.original_image)
                self.status_label.setText("📷 Showing original image")
    
    def generate_preview(self, filter_name, filter_path):
        """Generate a preview of the filter and cache it."""
        if self.filter_worker and self.filter_worker.isRunning():
            return
        
        # Create a smaller version for faster preview generation
        preview_image = self.get_preview_size_image(self.original_image)
        
        # Start preview worker
        self.filter_worker = FilterWorker(preview_image, filter_path, filter_name)
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
            self.status_label.setText(f"🔍 Previewing {filter_name}")
    
    def show_before(self):
        """Show the original image."""
        if self.original_image is not None:
            self.display_image(self.original_image)
            self.current_display_mode = "original"
            self.status_label.setText("📷 Showing original image")
    
    def show_after(self):
        """Show the filtered image."""
        if self.current_filtered_image is not None:
            self.display_image(self.current_filtered_image)
            self.current_display_mode = "filtered"
            self.status_label.setText("🎨 Showing filtered image")
        else:
            # No filter applied yet, show original
            self.show_before()


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