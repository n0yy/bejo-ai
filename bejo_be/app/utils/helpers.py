import hashlib
import json
from typing import Any, Dict, List, Optional
from datetime import datetime


def generate_content_hash(content: str) -> str:
    """Generate MD5 hash for content deduplication"""
    return hashlib.md5(content.encode()).hexdigest()


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime to ISO string"""
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove excessive whitespace
    text = " ".join(text.split())
    # Remove special characters that might interfere with processing
    text = text.replace("\x00", "").replace("\ufffd", "")
    return text.strip()


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string with fallback"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata from filename"""
    import os

    name, ext = os.path.splitext(filename)

    return {
        "filename": filename,
        "name": name,
        "extension": ext.lower(),
        "size": os.path.getsize(filename) if os.path.exists(filename) else 0,
        "modified_time": (
            format_timestamp(datetime.fromtimestamp(os.path.getmtime(filename)))
            if os.path.exists(filename)
            else None
        ),
    }
