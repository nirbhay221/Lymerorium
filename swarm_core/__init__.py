import sys
import os

# Allow bare imports inside swarm_core files (e.g. "import agents") to resolve correctly
sys.path.insert(0, os.path.dirname(__file__))
