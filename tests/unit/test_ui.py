import html
import json
import re

from pydantic import SecretStr

from mosaic.config import Settings
from mosaic.events.reporter import RunEvent
from mosaic.ui import replay
from mosaic.ui.art import TEAM, team_lineup, upload_icon
from mosaic.ui.office import office_actions, office_svg
from mosaic.ui.reviews import review_record, save_review


def ev(kind, title, t=0.0):
    return RunEvent(ts=t, kind=kind, title=title)


def test_run_events_become_office_scenes():
    events = [
        ev("step", "Reading your data"),
        ev("agent", "Dataset Triage Lead is working"),
        ev("agent", "Triage finished"),
        ev("agent", "Cleaning Strategist is working"),
        ev("agent", "Insight Analyst is working"),
        ev("guardrail", "Findings failed the fact check"),
        ev("fix", "Revised output accepted"),
        ev("agent", "Senior Reviewer is working"),
        ev("review", "Reviewer requested 1 change(s)"),
        ev("agent", "Insight Analyst is working"),
        ev("review", "Reviewer approved the findings"),
        ev("agent", "Report Writer is working"),
        ev("step", "Report ready"),
    ]
    actions = office_actions(events, "coffee")
    kinds = [(a["type"], a.get("who") or f"{a['from']}->{a['to']}") for a in actions]
    assert kinds[:4] == [
        ("deliver", "Tilly"),
        ("work", "Tilly"),
        ("handoff", "Tilly->Mop"),
        ("work", "Mop"),
    ]
    assert ("reject", "Pip") in kinds and ("say", "Pip") in kinds
    assert ("handoff", "Rex->Pip") in kinds  # a review asking for changes walks back to Pip
    assert ("approve", "Rex") in kinds and kinds[-1] == ("finale", "Quill")
    assert [a["id"] for a in actions] == list(range(len(actions)))
    assert office_actions([ev("error", "Run stopped")], "none")[0]["type"] == "fail"
    # the special analysts play Pip's part
    video = office_actions([ev("agent", "Video Synthesizer is working")], "none")
    assert video[0]["who"] == "Pip"


def test_the_office_and_welcome_scenes_have_the_whole_team():
    svg = office_svg()
    for m in TEAM:
        assert f'id="m-{m.name}"' in svg
    spots = json.loads(html.unescape(re.search(r'data-spots="([^"]+)"', svg).group(1)))
    assert set(spots) == {m.name for m in TEAM} | {"lounge", "gather"}
    assert len(spots["gather"]) == len(TEAM)
    lineup = team_lineup()
    assert all(m.name in lineup and m.role in lineup for m in TEAM)
    assert "done" in upload_icon(done=True) and "done" not in upload_icon()


def test_replays_round_trip_and_leave_time_for_each_scene(tmp_path):
    events = [ev("step", "Reading", 0), ev("agent", "Dataset Triage Lead is working", 0.5)]
    events += [ev("step", "Profiled", 1.0), ev("agent", "Report Writer is working", 60)]
    report = tmp_path / "report.html"
    report.write_text("<h1>r</h1>", encoding="utf-8")
    replay.save_replay(
        tmp_path / "replays" / "demo",
        label="Demo run",
        events=events,
        counters={"model_calls": 3},
        results_md="## Done",
        tiles="",
        charts=[],
        files=[str(report)],
    )
    assert replay.available(tmp_path / "replays") == [("demo", "Demo run")]
    data = replay.load("demo", tmp_path / "replays")
    assert data["events"][1].title == "Dataset Triage Lead is working"
    assert data["files"][0].endswith("report.html")
    times = replay.schedule(data["events"])
    assert times[1] >= replay.MIN_ACT_S  # enough time to act out the scene
    assert times[3] - times[2] <= max(replay.MAX_GAP_S, replay.MIN_ACT_S)  # no long waits
    for bad in ("../x", "", "missing"):
        try:
            replay.load(bad, tmp_path / "replays")
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should be rejected")


class FakeHub:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs["repo_id"], kwargs["path_in_repo"]))


def test_reviews_are_private_and_minimal(tmp_path):
    record = review_record(5, "  Lovely office!  " + "x" * 3000, " Sam ", {"finished": True})
    assert record["comment"].startswith("Lovely office!") and len(record["comment"]) == 2000
    assert record["name"] == "Sam" and set(record) == {
        "time",
        "stars",
        "comment",
        "name",
        "context",
    }

    local = Settings(_env_file=None, workspace_root=tmp_path)
    assert save_review(local, record) == "local"
    assert json.loads((tmp_path / "reviews.jsonl").read_text(encoding="utf-8"))["stars"] == 5

    hub = FakeHub()
    remote = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        hf_token=SecretStr("t"),
        reviews_repo="someone/mosaic-eda-reviews",
    )
    assert save_review(remote, record, api=hub) == "hub"
    assert [c[0] for c in hub.calls] == ["upload_file"]  # the private repo already exists
    assert hub.calls[0][1] == "someone/mosaic-eda-reviews"
    assert hub.calls[0][2].startswith("reviews/") and hub.calls[0][2].endswith(".json")


def test_the_app_builds_and_colors_can_switch_for_color_blind_viewers(monkeypatch, tmp_path):
    from mosaic.ui import app

    monkeypatch.setattr(replay, "REPLAY_DIR", tmp_path)
    fig = {"data": [{"type": "bar", "marker": {"color": "#2F6F73"}}], "layout": {}}
    assert app.recolor(fig)["data"][0]["marker"]["color"] == "#0072B2"
    demo = app.build_app()
    assert demo is not None


def test_the_agents_conversation_shows_who_said_what():
    from mosaic.ui.app import conversation_html, to_message

    msg = RunEvent(
        ts=0,
        kind="message",
        title="[table] Senior Reviewer to Insight Analyst: please revise",
        detail="- Finding 1: <b>too strong</b>. Please: soften it.",
        data={"sender": "Senior Reviewer", "recipient": "Insight Analyst"},
    )
    page = conversation_html([ev("step", "Profile ready"), msg])
    assert page.count('class="msg"') == 1 and "Rex" in page and "Pip" in page
    assert "please revise" in page and "table" in page
    assert "&lt;b&gt;too strong&lt;/b&gt;" in page  # the data's text is shown, not rendered
    assert "**Rex** (Senior Reviewer) to **Pip** (Insight Analyst)" in to_message(msg)["content"]
