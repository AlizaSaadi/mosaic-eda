"""Visitor reviews: stars, a comment, and an optional name, saved privately.

Each review is one small JSON file in a private Hugging Face Dataset repo that only the
owner can read (REVIEWS_REPO, written with HF_TOKEN). The repo must already exist: a
token limited to certain repos can write to them but not create new ones. Nothing is
emailed and no contact details are asked for. Without that setup, reviews are appended
to a local file instead.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from mosaic.config import Settings

MAX_COMMENT = 2000


def review_record(stars: int, comment: str, name: str, context: dict[str, Any]) -> dict:
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stars": int(stars),
        "comment": (comment or "").strip()[:MAX_COMMENT],
        "name": (name or "").strip()[:80],
        "context": context,  # the data type and whether the run succeeded, nothing more
    }


def save_review(settings: Settings, record: dict, api: Any = None) -> str:
    """Save one review. Returns where it went ('hub' or 'local')."""
    token = settings.hf_token.get_secret_value() if settings.hf_token else ""
    repo = settings.reviews_repo
    if token and repo:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        name = f"reviews/{record['time'][:10]}/{uuid.uuid4().hex[:12]}.json"
        api.upload_file(
            path_or_fileobj=json.dumps(record, indent=1).encode("utf-8"),
            path_in_repo=name,
            repo_id=repo,
            repo_type="dataset",
            commit_message="Add a visitor review",
        )
        return "hub"
    path = Path(settings.workspace_root) / "reviews.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return "local"
