"""
ui/slider_row.py - A reusable label + slider + value label row.
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, pyqtSignal

_SLIDER_STYLE = """
    QSlider::groove:horizontal {
        border: none;
        height: 2px;
        background: #cccccc;
        border-radius: 1px;
    }
    QSlider::handle:horizontal {
        background: #ffffff;
        border: 1px solid #999999;
        width: 12px;
        height: 12px;
        margin: -5px 0;
        border-radius: 6px;
    }
    QSlider::handle:horizontal:hover  { border-color: #666666; }
    QSlider::handle:horizontal:pressed {
        background: #f0f0f0;
        border-color: #333333;
    }
    QSlider::sub-page:horizontal {
        background: #999999;
        border-radius: 1px;
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
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(12)

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
        self._value_label.setAlignment(Qt.AlignCenter)
        self._value_label.setFixedWidth(30)
        self._value_label.setStyleSheet(
            "font-size:11px; color:#aaaaaa; background:transparent; border:none;"
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
