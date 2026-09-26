"""Save a copy of a report to a Hugging Face Dataset repo and return share links.

Opt-in per run: the user ticks a box, and only the HTML and PDF reports are uploaded
(never the uploaded data, the cleaned data, or the trace). Reports quote data only as
short, PII-masked snippets. The repo is public, so the links work for anyone.

Hugging Face serves uploaded HTML as plain text (a deliberate safety measure), so the
interactive report is shown by the Space itself: /?report=<job id> fetches it from the
dataset and displays it in a sandboxed frame. The PDF link opens directly in a browser.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any

from mosaic.config import Settings

SHARED_FILES = ("report.html", "report.pdf")
JOB_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")


class ShareUnavailable(RuntimeError):
    """Sharing isn't configured (no HF token or reports repo)."""


def sharing_configured(settings: Settings) -> bool:
    return bool(
        settings.hf_token and settings.hf_token.get_secret_value() and settings.reports_repo
    )


def viewer_url(job_id: str) -> str | None:
    """The Space's own link to a shared report (None when not running on a Space)."""
    host = os.environ.get("SPACE_HOST")
    return f"https://{host}/?report={job_id}" if host else None


def share_report(settings: Settings, job_id: str, out: Path, api: Any = None) -> dict[str, str]:
    """Upload the job's reports in one commit. Returns {link name: public link}."""
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
    links: dict[str, str] = {}
    viewer = viewer_url(job_id)
    if viewer and (out / "report.html").exists():
        links["Interactive report"] = viewer
    if (out / "report.pdf").exists():
        links["PDF report"] = f"{base}/report.pdf"
    return links


def fetch_shared_report(settings: Settings, job_id: str, download: Any = None) -> str:
    """The HTML of a shared report, from the public reports repo."""
    if not JOB_ID.match(job_id or "") or not settings.reports_repo:
        raise ValueError("That isn't a valid report link.")
    if download is None:
        from huggingface_hub import hf_hub_download as download
    path = download(
        repo_id=settings.reports_repo,
        repo_type="dataset",
        filename=f"reports/{job_id}/report.html",
    )
    return Path(path).read_text(encoding="utf-8")


def report_frame(report_html: str) -> str:
    """Show a report in a sandboxed frame: its charts' scripts run, but it can't touch the
    app, open pop-ups, or submit forms."""
    return (
        '<iframe title="Shared MOSAIC EDA report" sandbox="allow-scripts" '
        'style="width:100%;height:85vh;border:0;border-radius:12px" '
        f'srcdoc="{html.escape(report_html, quote=True)}"></iframe>'
    )
