"""
tag_rules.py

Regex-based rule definitions for transcription error tagging.

All Step 1 rules are expressed as compiled regular expressions so that
precise character spans can be recorded for each detected issue.

Later stages (Step 2 / Step 3) will extend this module.
"""

import regex as re


# ----------------------------------------------------------------------
# Step 1 regex rules
# ----------------------------------------------------------------------

LEADING_WHITESPACE = re.compile(r"(?m)^[ \t]+")

TRAILING_WHITESPACE = re.compile(r"(?m)[ \t]+$")

INTERNAL_WHITESPACE = re.compile(r"[ \t]{2,}")

SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+[.,;:!?]")

SUSPICIOUS_UNICODE = re.compile(r"[\u200B-\u200D\uFEFF]")

REPEATED_PUNCTUATION = re.compile(r"[.,;:!?]{2,}")

NON_LATIN_GLYPH = re.compile(r"[^\p{Latin}\p{N}\p{P}\p{Zs}]")

MALFORMED_CHARACTER = re.compile(r"\uFFFD")

MIXED_PUNCTUATION = re.compile(r"\w[.,;:!?]\w")


# ----------------------------------------------------------------------
# Aggregate Step 1 tags
# ----------------------------------------------------------------------

all_step1_tags = {
    "L": LEADING_WHITESPACE,
    "T": TRAILING_WHITESPACE,
    "W": INTERNAL_WHITESPACE,
    "SP": SPACE_BEFORE_PUNCTUATION,
    "C": SUSPICIOUS_UNICODE,
    "P": REPEATED_PUNCTUATION,
    "G": NON_LATIN_GLYPH,
    "M": MALFORMED_CHARACTER,
    "MP": MIXED_PUNCTUATION,
}
