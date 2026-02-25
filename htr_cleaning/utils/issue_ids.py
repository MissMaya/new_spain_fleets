"""
issue_ids.py

Utility functions for issuing IDs.

These IDs are stable across reruns and depend only on:
    - doc_id
    - tag
    - _abs_start
    - _abs_end
"""

import hashlib


def generate_issue_id(doc_id: str, tag: str, start: int, end: int) -> str:
    """
    Deterministic hash-based issue ID.

    Parameters:
        doc_id (str): Document identifier
        tag (str): Issue tag (e.g., S2X)
        start (int): Absolute start offset
        end (int): Absolute end offset

    Returns:
        str: 12-character stable hash ID
    """
    base = f"{doc_id}|{tag}|{start}|{end}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]