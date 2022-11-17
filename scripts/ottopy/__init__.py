from ctypes import cdll
from pathlib import Path

LIBOTTO_SO_PATH = Path(__file__).resolve().parent.parent.parent / "build" / "libottopy.so"

if not LIBOTTO_SO_PATH.exists():
    raise FileNotFoundError("Could not find '.so' file at '{}'. Are you sure you built the lib?".format(
        str(LIBOTTO_SO_PATH)
    ))


lib = cdll.LoadLibrary(str(LIBOTTO_SO_PATH))
