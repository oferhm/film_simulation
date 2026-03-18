"""
ui/slider_row.py - A reusable label + slider + value label row.
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, pyqtSignal

_SLIDER_STYLE = """
    QSlider::groove:horizontal {
        border: none;
        height: 1px;
        background: #404040;
        border-radius: 0px;
    }
    QSlider::handle:horizontal {
        background: #ffffff;
        border: none;
        width: 8px;
        height: 8px;
        margin: -3px 0;
        border-radius: 4px;
    }
    QSlider::handle:horizontal:hover { 
        background: #e0e0e0;
    }
    QSlider::handle:horizontal:pressed {
        background: #cccccc;
    }
    QSlider::sub-page:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                   stop:0 #ffffff, 
                                   stop:0.3 #cccccc, 
                                   stop:0.7 #808080, 
                                   stop:1 #404040);
        border-radius: 0px;
    }
"""


class SliderRow(QWidget):
    """
    Horizontal row: label | slider | numeric value.

    Signals
    -------
    valueChanged(int)   – emitted on every slider move
    doubleClicked()     – emitted when the slider is double-clicked (reset hint)
    """

    valueChanged  = pyqtSignal(int)
    doubleClicked = pyqtSignal()

    def __init__(
        self,
        label: str,
        minimum: int = -100,
        maximum: int = 100,
        default: int = 0,
        label_width: int = 70,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet("border: none; background: transparent;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(16)

        # Label
        lbl = QLabel(label)
        lbl.setFixedWidth(label_width)
        lbl.setStyleSheet(
            "font-size:12px; font-weight:normal; color:#ffffff;"
            "background:transparent; border:none;"
        )
        layout.addWidget(lbl)

        # Slider
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(minimum)
        self._slider.setMaximum(maximum)
        self._slider.setValue(default)
        self._slider.setTracking(True)
        self._slider.setStyleSheet(_SLIDER_STYLE)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_value_changed)
        self._slider.mouseDoubleClickEvent = self._on_double_click
        layout.addWidget(self._slider)

        # Value label
        self._value_label = QLabel(str(default))
        self._value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._value_label.setFixedWidth(35)
        self._value_label.setStyleSheet(
            "font-size:12px; color:#ffffff; background:transparent; border:none;"
        )
        layout.addWidget(self._value_label)

    # ── public API ───────────────────────────────────────────────────────────

    @property
    def value(self) -> int:
        return self._slider.value()

    def set_value(self, v: int):
        self._slider.setValue(v)

    def set_enabled(self, enabled: bool):
        self._slider.setEnabled(enabled)

    def reset(self):
        self._slider.setValue(0)

    # ── private ──────────────────────────────────────────────────────────────

    def _on_value_changed(self, v: int):
        self._value_label.setText(str(v))
        self.valueChanged.emit(v)

    def _on_double_click(self, event):
        if event.button() == Qt.LeftButton:
            self.reset()
            self.doubleClicked.emit()
