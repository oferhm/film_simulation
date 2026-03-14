#!/usr/bin/env python3
"""
Launch the Film Filter GUI Application
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from film_filter_gui import main
    main()
except ImportError as e:
    print("Error: Missing dependencies. Please install required packages:")
    print("pip install PyQt5")
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error launching GUI: {e}")
    sys.exit(1)