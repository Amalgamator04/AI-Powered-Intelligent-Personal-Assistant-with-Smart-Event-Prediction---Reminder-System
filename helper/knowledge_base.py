import os
from datetime import datetime


def append_to_kb(text, file_path="knowledge_base.txt", add_timestamp=True):
    """Append `text` to `file_path`. Create the file if it doesn't exist.

    Each entry is written on its own line with an optional timestamp.
    Returns a tuple `(abs_path, written_line)` when text was written, or None when
    `text` is empty or falsy.
    """
    if not text:
        return None

    # Ensure parent directory exists (works whether file_path is relative or absolute)
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if add_timestamp else None
    written_line = f"[{ts}] {text}" if ts else text

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(written_line + "\n")
        f.flush()

    abs_path = os.path.abspath(file_path)
    print(f"Text added to knowledge base ({abs_path})")
    return abs_path, written_line

