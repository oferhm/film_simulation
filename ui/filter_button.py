"""
ui/filter_button.py - Custom QPushButton with hover-preview and selection state.
"""

from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QTimer, pyqtSignal

_BASE_STYLE = """
    QPushButton {{
        background-color: {bg_color};
        border: none;
        border-radius: 0px;
        padding: {padding};
        font-size: 12px;
        font-weight: {weight};
        color: {text_color};
        text-align: left;
        margin: 0px;
    }}
    QPushButton:hover {{
        background-color: {hover_bg};
        border-color: {hover_border};
    }}
    QPushButton:pressed {{ background-color: #094771; }}
    QPushButton:disabled {{
        background-color: #252526;
        color: #656565;
        border-color: #3e3e42;
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
        self.setFixedHeight(24)  # More compact like VS Code
        self.setContentsMargins(0, 0, 0, 0)

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
                bg_color="#094771",
                padding="5px 12px",
                weight="400",
                text_color="#cccccc",
                hover_bg="#1177bb",
                hover_border="#0078d4",
            ))
        else:
            self.setStyleSheet(_BASE_STYLE.format(
                bg_color="#383838",
                padding="5px 12px",
                weight="400",
                text_color="#cccccc",
                hover_bg="#2a2d2e",
                hover_border="#464647",
            ))
