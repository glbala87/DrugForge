"""pytest configuration — add project root to sys.path."""

import os
import sys

# Add project root to path so tests can import modules directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
