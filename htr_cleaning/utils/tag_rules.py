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
NON_LATIN_GLYPH = re.compile(r"[^\p{Latin}\p{N}\p{P}\p{Zs}]")
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

# Q not followed by E or I
QU_NOT_EI = re.compile(r"QU(?![EI])", re.IGNORECASE)

# Presence of W or K
W_OR_K = re.compile(r"[WK]", re.IGNORECASE)

# Unexpected double consonants (excluding cc, ll, nn, rr)
DOUBLE_CONSONANT = re.compile(
    r"(?i)(?<!c)c{2}|(?<!l)l{2}|(?<!n)n{2}|(?<!r)r{2}|"  # guards
    r"(bb|dd|ff|gg|hh|jj|kk|mm|pp|qq|ss|tt|vv|ww|xx|yy|zz)"
)

# Rare final consonants: C F K M P
RARE_FINAL_CONSONANT = re.compile(r"(?i)\b\w+[CFKMP]\b")

# Triple letter repetition
TRIPLE_LETTER = re.compile(r"(?i)([a-z])\1\1")

all_step3_tags = {
    "Q": QU_NOT_EI,
    "WK": W_OR_K,
    "DC": DOUBLE_CONSONANT,
    "E": RARE_FINAL_CONSONANT,
    "T": TRIPLE_LETTER,
}
