# Configuration for pytest
# This file can contain fixtures and configurations for testing

import pytest
import sys
import os

# Add the backend/src directory to the Python path so tests can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))