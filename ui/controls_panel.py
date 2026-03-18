"""
ui/controls_panel.py - Right-hand panel: adjustment sliders + filter list.
"""

import os
# Import QtWidgets components
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QScrollArea, QWidget, QGroupBox, QLabel
# Import QtCore components including QTimer
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from ui.slider_row import SliderRow
from ui.filter_button import FilterButton
from core.state import EditState

# Ordered list of (field_name, display_label)
_SLIDERS: list[tuple[str, str]] = [
    ("temp",       "Temp"),
    ("tint",       "Tint"),
    ("exposure",   "Exposure"),
    ("contrast",   "Contrast"),
    ("highlights", "Highlights"),
    ("shadows",    "Shadows"),
    ("whites",     "Whites"),
    ("blacks",     "Blacks"),
    ("texture",    "Texture"),
    ("clarity",    "Clarity"),
    ("dehaze",     "Dehaze"),
    ("vibrance",   "Vibrance"),
    ("saturation", "Saturation"),
]


class ControlsPanel(QFrame):
    """
    Signals
    -------
    editChanged(EditState)          – any slider moved
    filterSelected(name, path)      – filter button clicked (select)
    filterDeselected()              – same filter clicked again
    filterHovered(name, path)       – mouse entered a filter button
    filterHoverLeft()               – mouse left a filter button
    """

    editChanged     = pyqtSignal(object)   # EditState
    filterSelected  = pyqtSignal(str, str)
    filterDeselected = pyqtSignal()
    filterHovered   = pyqtSignal(str, str)
    filterHoverLeft = pyqtSignal()

    def __init__(self, filters_dir: str, parent=None):
        super().__init__(parent)
        self.setObjectName("controlsPanel")
        self.setStyleSheet("""
            #controlsPanel {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 12px;
            }
        """)

        self._filters_dir = filters_dir
        self._edit_state  = EditState()
        self._selected_filter: str | None = None
        self._selected_btn: FilterButton | None = None
        self._filter_buttons: list[FilterButton] = []

        # Debounce timers: one per slider field
        self._timers: dict[str, QTimer] = {}
        for field, _ in _SLIDERS:
            t = QTimer(singleShot=True)
            t.timeout.connect(lambda f=field: self._emit_edit(f))
            self._timers[field] = t

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(4)

        # Sliders
        self._slider_rows: dict[str, SliderRow] = {}
        for field, label in _SLIDERS:
            row = SliderRow(label)
            row.valueChanged.connect(lambda v, f=field: self._on_slider_changed(f, v))
            layout.addWidget(row)
            self._slider_rows[field] = row

        # Filter group
        group = QGroupBox("Filters")
        group.setCheckable(True)
        group.setChecked(True)
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px; font-weight: 600; color: #ffffff;
                background: transparent;
                border: 1px solid #404040; border-radius: 8px;
                margin-top: 10px; padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px;
                padding: 0 8px; background: #2d2d2d;
            }
        """)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setWidgetResizable(True)
        scroll.verticalScrollBar().setSingleStep(10)

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(0)

        available = self._discover_filters()
        if not available:
            lbl = QLabel("No filters found")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#888888; font-size:14px; padding:40px;")
            inner_layout.addWidget(lbl)
        else:
            for name, path in available.items():
                btn = FilterButton(name, path)
                btn.setEnabled(False)
                btn.clicked.connect(
                    lambda _, n=name, p=path, b=btn: self._on_filter_clicked(n, p, b)
                )
                btn.hoverEntered.connect(self.filterHovered)
                btn.hoverLeft.connect(self.filterHoverLeft)
                inner_layout.addWidget(btn)
                self._filter_buttons.append(btn)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        group_layout.addWidget(scroll)
        layout.addWidget(group)

    # ── public API ───────────────────────────────────────────────────────────

    def set_controls_enabled(self, enabled: bool):
        for row in self._slider_rows.values():
            row.set_enabled(enabled)
        for btn in self._filter_buttons:
            btn.setEnabled(enabled)

    def reset_all(self):
        """Reset every slider to 0 without emitting editChanged."""
        self._edit_state = EditState()
        for row in self._slider_rows.values():
            row.blockSignals(True)
            row.reset()
            row.blockSignals(False)

    def deselect_filter(self):
        if self._selected_btn:
            self._selected_btn.set_selected(False)
        self._selected_filter = None
        self._selected_btn = None

    @property
    def edit_state(self) -> EditState:
        return self._edit_state

    # ── private ──────────────────────────────────────────────────────────────

    def _discover_filters(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if not os.path.isdir(self._filters_dir):
            return result
        for fname in sorted(os.listdir(self._filters_dir)):
            if fname.endswith(".py") and not fname.startswith("__"):
                display = (
                    fname.replace(".py", "")
                         .replace("_", " ")
                         .title()
                         .replace(" Copy", "")
                )
                result[display] = os.path.join(self._filters_dir, fname)
        return result

    def _on_slider_changed(self, field: str, value: int):
        self._edit_state = self._edit_state.with_field(field, value)
        # Debounce: restart the timer for this field
        self._timers[field].start(50)

    def _emit_edit(self, _field: str):
        self.editChanged.emit(self._edit_state)

    def _on_filter_clicked(self, name: str, path: str, btn: FilterButton):
        if self._selected_filter == name:
            self.deselect_filter()
            self.filterDeselected.emit()
        else:
            if self._selected_btn:
                self._selected_btn.set_selected(False)
            self._selected_filter = name
            self._selected_btn = btn
            btn.set_selected(True)
            self.filterSelected.emit(name, path)
