"""
tag_rules.py

Regex-based rules for transcription error tagging in steps 1 and 3.

Step 1:
- Basic anomalies (whitespace, repeated punctuation, glyphs)

Step 3:
- Linguistic / paleographic heuristics

All rules are expressed as compiled regular expressions so that
precise character spans can be recorded for each detected issue.

Step 2 tagging (character alignment) can be found under utils/alignment.py.
"""

import regex as re


# ----------------------------------------------------------------------
# STEP 1 – Surface anomalies
# ----------------------------------------------------------------------

LEADING_WHITESPACE = re.compile(r"(?m)^[ \t]+")
TRAILING_WHITESPACE = re.compile(r"(?m)[ \t]+$")
INTERNAL_WHITESPACE = re.compile(r"[ \t]{2,}")
SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+[.,;:!?]")
SUSPICIOUS_UNICODE = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]")
REPEATED_PUNCTUATION = re.compile(r"[.,;:!?]{2,}")
NON_LATIN_GLYPH = re.compile(r"[^\p{Latin}\p{N}\p{P}\p{Zs}\n\r\t]")
MALFORMED_CHARACTER = re.compile(r"\uFFFD")
MIXED_PUNCTUATION = re.compile(r"\w[.,;:!?]\w")

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


# ----------------------------------------------------------------------
# STEP 3 – Linguistic / paleographic heuristics
# ----------------------------------------------------------------------

# QU not followed by E or I
QU_NOT_EI = re.compile(r"(?i)\bqu(?![ei])")

# Presence of W or K
W_OR_K = re.compile(r"(?i)[wk]")

# Unexpected doubled consonants excluding cc, ll, nn, rr
DOUBLE_CONSONANT = re.compile(r"(?i)([bdfghjkmmpqstvwxyz])\1")

# Rare final consonants: C F K M P
RARE_FINAL_CONSONANT = re.compile(r"(?i)\b\w+[cfkmp]\b")

# Triple letter repetition
TRIPLE_LETTER = re.compile(r"(?i)([a-z])\1\1")

all_step3_tags = {
    "Q": QU_NOT_EI,
    "WK": W_OR_K,
    "DC": DOUBLE_CONSONANT,
    "E": RARE_FINAL_CONSONANT,
    "T": TRIPLE_LETTER,
}


