import logging
import os
from typing import List


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def validate_required_env(keys: List[str]) -> List[str]:
    missing = [k for k in keys if not os.getenv(k)]
    return missing
