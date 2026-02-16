"""
normalisation.py

Utility used to clean up confusion matrices by removing
certain characters e.g. the Herbrew accent characters that 
were detected in the first run and by normalising others e.g. 
collapsing the representation of multiple whitespace into a 
single space representation.
"""

import regex as re
import unicodedata

# Token for null characters 
NULL_TOKEN = "∅"

# Token for unknown glyphs
OTHER_TOKEN = "OTHER"


def normalise_char(char: str, lowercase: bool = False) -> str:
    """
    Normalise a single character for analytical comparison.

    Steps:
    - Replace null characters with the ∅ token
    - Remove zero-width characters
    - Remove control characters
    - Collapse whitespace to single space
    - Optionally lowercase
    - Replace non-Latin / exotic glyphs with the word OTHER
    """

    if char is None:
        return NULL_TOKEN

    # Explicit NULL handling
    if char == "":
        return NULL_TOKEN

    # Remove zero-width characters
    if re.match(r"[\u200B-\u200D\uFEFF]", char):
        return ""

    # Remove control characters
    if unicodedata.category(char).startswith("C"):
        return ""

    # Normalize whitespace
    if char.isspace():
        return " "

    # Retain Latin letters, numbers, and punctuation
    if re.match(r"[\p{Latin}\p{N}\p{P}\p{Zs}]", char):
        return char

    # Everything else gets marked as OTHER e.g. the previously detected Hebrew accents
    return OTHER_TOKEN


def normalise_pair(gt_char: str, htr_char: str, lowercase: bool = False):
    """
    Normalise a pair of aligned characters.
    """

    g = normalise_char(gt_char, lowercase)
    h = normalise_char(htr_char, lowercase)

    if g == "":
        g = NULL_TOKEN
    if h == "":
        h = NULL_TOKEN

    return g, h