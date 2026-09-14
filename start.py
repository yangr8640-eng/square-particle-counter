"""Isolated runtime entry point for the local desktop distribution."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_app.__main__ import main
if __name__ == '__main__':
    main()
