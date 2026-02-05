"""
tagging_rendering.py

Render detected issues (Steps 1–3) inline into HTR transcripts for human review.

Output:
    data/tagged/<style>/<doc_id>.txt

Uses:
    logs/<style>/<doc_id>/issues_with_ids.json if present,
    otherwise falls back to issues.json.

Each opening tag includes a human-readable description pulled from:

    schemas_and_manifests/tag_schema.json

Example:

    <S3:Q desc="QU not followed by E or I" id="...">Q</S3>

If multilingual descriptions are provided in the schema, English ("en") is used
by default.

Tags are injected by character spans.
Deletions (start == end) are rendered as Ø.

Rendering is deterministic and preserves original HTR content.
Raw HTR files are NEVER modified.
"""

from pathlib import Path
from typing import List, Dict, Tuple

from utils.file_io import read_text, load_json_if_exists, safe_write_text
from utils.alignment import NULL_CHAR
from utils.config import PROJECT_ROOT


# ---------------------------------------------------------------------
# Load tag schema once
# ---------------------------------------------------------------------

TAG_SCHEMA_PATH = PROJECT_ROOT / "schemas_and_manifests" / "tag_schema.json"
TAG_SCHEMA = load_json_if_exists(TAG_SCHEMA_PATH, {})

S1_TAGS = {"L", "T", "W", "SP", "C", "P", "G", "M", "MP"}
S2_TAGS = {"X", "I", "D"}


def _infer_step(tag: str) -> str:
    if tag in S1_TAGS:
        return "S1"
    if tag in S2_TAGS:
        return "S2"
    return "S3"


def _lookup_description(step: str, tag: str) -> str:
    """
    Retrieve human-readable description from tag_schema.json.

    Supports either:
        "Q": "description"
    or:
        "Q": {"en": "...", "es": "..."}
    """

    step_block = TAG_SCHEMA.get(step, {})
    entry = step_block.get(tag, "")

    if isinstance(entry, dict):
        # Prefer English if available, otherwise first value
        return entry.get("en") or next(iter(entry.values()), "")
    elif isinstance(entry, str):
        return entry

    return ""


def _make_tokens(issue: Dict):
    tag = issue.get("tag")
    step = issue.get("step") or _infer_step(tag)
    issue_id = issue.get("issue_id")

    desc = _lookup_description(step, tag)
    desc_attr = f' desc="{desc}"' if desc else ""

    if issue_id:
        open_tag = f'<{step}:{tag}{desc_attr} id="{issue_id}">'
    else:
        open_tag = f"<{step}:{tag}{desc_attr}>"

    close_tag = f"</{step}>"

    return open_tag, close_tag


def render_tags_for_document(htr_text: str, issues: List[Dict]) -> str:
    """
    Insert tags into HTR text using span positions.

    Issues must contain:
        - tag
        - start
        - end
    Optional:
        - issue_id
        - step
    """

    inserts: List[Tuple[int, int, str]] = []
    # (position, priority, token)
    # priority ensures close tags at same position are applied before open tags

    for issue in issues:
        start = int(issue["start"])
        end = int(issue["end"])

        open_tag, close_tag = _make_tokens(issue)

        if start == end:
            # deletion / anchor
            inserts.append((start, 1, f"{open_tag}{NULL_CHAR}{close_tag}"))
        else:
            inserts.append((end, 0, close_tag))
            inserts.append((start, 1, open_tag))

    # Insert from end backwards so offsets remain valid
    inserts.sort(key=lambda x: (x[0], x[1]), reverse=True)

    tagged = htr_text
    for pos, _, token in inserts:
        pos = max(0, min(len(tagged), pos))
        tagged = tagged[:pos] + token + tagged[pos:]

    return tagged


def render_all_tagged_transcripts(
    train_pairs,
    logs_dir: Path,
    output_root: Path,
):
    """
    Render tagged HTR files for provided document pairs.

    NOTE:
        Caller controls whether these are TRAIN or TEST pairs.
        In pipeline usage this should be TEST only.

    Writes:
        output_root/<style>/<doc_id>.txt
    """

    for pair in train_pairs:
        style = pair["style"]
        doc_id = pair["id"]
        htr_path = Path(pair["htr_path"])

        doc_log_dir = logs_dir / style / doc_id

        issues_with_ids = doc_log_dir / "issues_with_ids.json"
        issues_plain = doc_log_dir / "issues.json"

        issues = load_json_if_exists(issues_with_ids, None)
        if issues is None:
            issues = load_json_if_exists(issues_plain, [])

        if not issues:
            continue

        htr_text = read_text(htr_path)
        tagged_text = render_tags_for_document(htr_text, issues)

        out_dir = output_root / style
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{doc_id}.txt"
        safe_write_text(tagged_text, out_path)
        