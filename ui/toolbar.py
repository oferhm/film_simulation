"""
ui/toolbar.py - Top toolbar: Import, path labels, Save As.
"""

from PyQt5.QtWidgets import QFrame, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import pyqtSignal

_BTN_STYLE = """
    QPushButton {{
        background-color: #383838;
        border: 1px solid #464647;
        border-radius: 3px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 400;
        color: #cccccc;
        text-align: {align};
    }}
    QPushButton:hover  {{ background-color: #2a2d2e; border-color: #464647; }}
    QPushButton:pressed {{ background-color: #094771; }}
    QPushButton:disabled {{
        background-color: #252526;
        color: #656565;
        border-color: #3e3e42;
    }}
"""

_PATH_LABEL_STYLE = """
    QLabel {
        color: #cccccc;
        font-size: 11px;
        padding: 4px 8px;
        background-color: transparent;
        max-width: 300px;
        min-width: 200px;
    }
"""


class TopToolbar(QFrame):
    """
    Signals
    -------
    importClicked()
    saveClicked()
    """

    importClicked = pyqtSignal()
    saveClicked   = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 8px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Import
        self._import_btn = QPushButton("Import Photos")
        self._import_btn.setFixedHeight(36)
        self._import_btn.setStyleSheet(_BTN_STYLE.format(align="left"))
        self._import_btn.clicked.connect(self.importClicked)
        layout.addWidget(self._import_btn)

        self._import_label = QLabel()
        self._import_label.setStyleSheet(_PATH_LABEL_STYLE)
        layout.addWidget(self._import_label)

        layout.addStretch(2)

        # Save
        self._save_btn = QPushButton("Save as")
        self._save_btn.setFixedHeight(36)
        self._save_btn.setEnabled(False)
        self._save_btn.setStyleSheet(_BTN_STYLE.format(align="center"))
        self._save_btn.clicked.connect(self.saveClicked)
        layout.addWidget(self._save_btn)

        self._save_label = QLabel()
        self._save_label.setStyleSheet(_PATH_LABEL_STYLE)
        layout.addWidget(self._save_label)

        layout.addStretch(2)

    # ── public API ───────────────────────────────────────────────────────────

    def set_import_path(self, path: str):
        self._import_label.setText(path)

    def set_save_path(self, path: str):
        self._save_label.setText(path)

    def set_save_enabled(self, enabled: bool):
        self._save_btn.setEnabled(enabled)
