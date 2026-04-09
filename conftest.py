# conftest.py  — place this in the project ROOT (same folder as main.py)
#
# Purpose: adds the project root to sys.path so pytest can always find
#          the 'agents' and 'main' modules regardless of which directory
#          pytest is launched from.

import sys
import os

# Insert the project root at the front of sys.path
sys.path.insert(0, os.path.dirname(__file__))
