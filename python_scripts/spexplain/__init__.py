from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


class Error(Exception):
    """An error to be reported to the user without a traceback."""
