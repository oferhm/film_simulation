"""
app.py - QApplication setup and bootstrap.
"""

import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Film Filter Studio")
    app.setOrganizationName("Film Filters")

    window = MainWindow()
    window.show()

    return app.exec_()
