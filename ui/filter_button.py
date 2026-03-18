"""
ui/filter_button.py - Custom QPushButton with hover-preview and selection state.
"""

from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QTimer, pyqtSignal

_BASE_STYLE = """
    QPushButton {{
        background-color: #3c3c3c;
        border: {border};
        border-radius: 8px;
        padding: {padding};
        font-size: 13px;
        font-weight: {weight};
        color: #ffffff;
        text-align: left;
    }}
    QPushButton:hover {{
        background-color: #4c4c4c;
        border-color: {hover_border};
    }}
    QPushButton:pressed {{ background-color: #2c2c2c; }}
    QPushButton:disabled {{
        background-color: #282828;
        color: #666666;
        border-color: #333333;
    }}
"""


class FilterButton(QPushButton):
    """
    A filter-list button that:
    - emits hoverEntered(name, path) after a short delay
    - emits hoverLeft()
    - tracks a selected state with distinct styling
    """

    hoverEntered = pyqtSignal(str, str)
    hoverLeft    = pyqtSignal()

    _HOVER_DELAY_MS = 50

    def __init__(self, filter_name: str, filter_path: str, parent=None):
        super().__init__(filter_name, parent)
        self.filter_name = filter_name
        self.filter_path = filter_path
        self.is_selected = False
        self.setFixedHeight(45)

        self._hover_timer = QTimer(singleShot=True)
        self._hover_timer.timeout.connect(self._emit_hover)

        self._apply_style()

    # ── public API ───────────────────────────────────────────────────────────

    def set_selected(self, selected: bool):
        self.is_selected = selected
        self._apply_style()

    # ── Qt events ────────────────────────────────────────────────────────────

    def enterEvent(self, event):
        super().enterEvent(event)
        self._hover_timer.start(self._HOVER_DELAY_MS)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if self._hover_timer.isActive():
            self._hover_timer.stop()
        else:
            self.hoverLeft.emit()

    # ── private ──────────────────────────────────────────────────────────────

    def _emit_hover(self):
        self.hoverEntered.emit(self.filter_name, self.filter_path)

    def _apply_style(self):
        if self.is_selected:
            self.setStyleSheet(_BASE_STYLE.format(
                border="2px solid #0078d4",
                padding="15px 16px",
                weight="600",
                hover_border="#106ebe",
            ))
        else:
            self.setStyleSheet(_BASE_STYLE.format(
                border="1px solid #555555",
                padding="16px",
                weight="500",
                hover_border="#777777",
            ))
