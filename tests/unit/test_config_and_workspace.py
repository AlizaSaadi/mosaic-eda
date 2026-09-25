import os
import time

from mosaic.config import Settings
from mosaic.workspace import SUBDIRS, create_workspace, sweep_stale


def test_pools_parse_from_comma_separated_env(monkeypatch):
    monkeypatch.setenv("GEMINI_LITE_POOL", "model-x, model-y")
    settings = Settings(_env_file=None)
    assert settings.gemini_lite_pool == ["model-x", "model-y"]


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert Settings(_env_file=None).has_gemini_key is False


def test_workspace_layout_and_cleanup(tmp_path):
    ws = create_workspace(tmp_path)
    for name in SUBDIRS:
        assert (ws.root / name).is_dir()
    ws.cleanup()
    assert not ws.root.exists()


def test_sweep_removes_only_stale_jobs(tmp_path):
    old = create_workspace(tmp_path, "old-job")
    fresh = create_workspace(tmp_path, "fresh-job")
    past = time.time() - 3 * 3600
    os.utime(old.root, (past, past))
    assert sweep_stale(tmp_path, ttl_minutes=60) == ["old-job"]
    assert fresh.root.exists()
