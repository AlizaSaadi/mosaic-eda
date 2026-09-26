"""Save a copy of a report to a Hugging Face Dataset repo and return share links.

Opt-in per run: the user ticks a box, and only the HTML and PDF reports are uploaded
(never the uploaded data, the cleaned data, or the trace). Reports quote data only as
short, PII-masked snippets. The repo is public, so the links work for anyone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mosaic.config import Settings

SHARED_FILES = ("report.html", "report.pdf")


class ShareUnavailable(RuntimeError):
    """Sharing isn't configured (no HF token or reports repo)."""


def sharing_configured(settings: Settings) -> bool:
    return bool(
        settings.hf_token and settings.hf_token.get_secret_value() and settings.reports_repo
    )


def share_report(settings: Settings, job_id: str, out: Path, api: Any = None) -> dict[str, str]:
    """Upload the job's reports in one commit. Returns {file name: public link}."""
    if not sharing_configured(settings):
        raise ShareUnavailable("Sharing isn't set up: add HF_TOKEN and REPORTS_REPO.")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=settings.hf_token.get_secret_value())
    from huggingface_hub import CommitOperationAdd

    repo = settings.reports_repo
    files = [out / name for name in SHARED_FILES if (out / name).exists()]
    if not files:
        raise ShareUnavailable("There's no report to share.")
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=f"reports/{job_id}/{f.name}", path_or_fileobj=str(f))
            for f in files
        ],
        commit_message=f"Add MOSAIC EDA report {job_id}",
    )
    base = f"https://huggingface.co/datasets/{repo}/resolve/main/reports/{job_id}"
    return {f.name: f"{base}/{f.name}" for f in files}
