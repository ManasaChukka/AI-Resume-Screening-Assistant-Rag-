import hashlib
import logging
import json
from pathlib import Path

LOG_DIR = Path("logs")
CACHE_DIR = Path("data/cache")


def setup_logger(name: str = "resume_rag") -> logging.Logger:
    """
    Configure and return a logger that writes to both console and file.

    Args:
        name: Name identifying this logger instance.

    Returns:
        A configured Logger object.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(LOG_DIR / "app.log")
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Compute a stable hash of file contents, used as a cache key.

    Args:
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        A hex string hash uniquely identifying this file's content.
    """
    return hashlib.md5(file_bytes).hexdigest()


def load_cache() -> dict:
    """
    Load the processed-files cache from disk.

    Returns:
        A dict mapping file hashes to their processed chunk metadata.
        Returns an empty dict if no cache file exists yet.
    """
    cache_file = CACHE_DIR / "processed_files.json"
    if not cache_file.exists():
        return {}
    with open(cache_file, "r") as f:
        return json.load(f)


def save_cache(cache: dict) -> None:
    """
    Persist the processed-files cache to disk.

    Args:
        cache: The cache dictionary to save.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "processed_files.json"
    with open(cache_file, "w") as f:
        json.dump(cache, f)
