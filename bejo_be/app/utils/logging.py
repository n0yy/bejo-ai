import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", debug: bool = False):
    """Setup application logging"""

    if debug:
        level = "DEBUG"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log", mode="a"),
        ],
    )

    # Set specific logger levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {level}")
