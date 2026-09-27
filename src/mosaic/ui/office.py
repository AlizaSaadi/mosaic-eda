"""The Mosaic office: a small SVG floor plan where the agents work while a job runs.

The layout is static. The live part is a list of office actions derived from the run's
events. Walks between rooms come only from the agents' messages to each other (each names
its sender and recipient), so the office shows the real conversation; "is working" events
only light up a desk. The browser script (static/office.js) polls the list from a hidden
textbox and plays each action once.
"""

from __future__ import annotations

import html
import json
from typing import Any

from mosaic.events.reporter import RunEvent
from mosaic.ui.art import AGENT_TO_MEMBER, TEAM, creature, drink

W, H = 960, 470
ROOM_W, ROOM_H = 290, 180
COLS = (16, 335, 654)
TOP_Y, BOTTOM_Y = 16, 276
CORRIDOR_Y = 236  # where walking creatures' feet are

# which room each member works in: (column, row)
ROOMS = {"Tilly": (0, 0), "Mop": (1, 0), "Pip": (2, 0), "Rex": (0, 1), "Quill": (1, 1)}
LOUNGE = (2, 1)

WORKING = {
    "Tilly": "Tilly is reading your data",
    "Mop": "Mop is planning the cleaning",
    "Pip": "Pip is looking for insights",
    "Rex": "Rex is reviewing the findings",
    "Quill": "Quill is writing your report",
}
# what a creature says when it hands over each kind of message (by the message's subject)
SUBJECT_LINES = {
    "triage brief": "Here's the brief!",
    "cleaning plan, applied": "Data's clean!",
    "findings for review": "Findings ready",
    "revised findings": "Revised!",
    "please revise": "Please revise",
    "out of revision rounds": "Out of rounds",
    "couldn't revise": "Couldn't fix it",
    "final findings to write up": "Write it up!",
}
# the office caption for each message ({s} sender, {r} recipient)
SUBJECT_CAPTIONS = {
    "triage brief": "{s} gives {r} the brief",
    "cleaning plan, applied": "{s} hands {r} the cleaned data",
    "findings for review": "{s} sends {r} the findings to review",
    "revised findings": "{s} sends {r} the revised findings",
    "please revise": "{s} sends the findings back to {r} with comments",
    "out of revision rounds": "{s} withholds the findings still in dispute",
    "couldn't revise": "{s} tells {r} the revision didn't pass the fact check",
    "final findings to write up": "{s} gives {r} the final findings to write up",
}
RED = {"please revise", "out of revision rounds"}
# the caption while a check sends work back, by who owns the stage
REDO_CAPTIONS = {
    "Tilly": "A check sent Tilly's brief back; she's correcting it",
    "Mop": "The dry run rejected Mop's plan; he's revising it",
    "Pip": "The fact check caught a number; Pip is fixing it",
    "Rex": "Rex's review wasn't consistent; he's redoing it",
    "Quill": "Quill is rewriting part of the summary",
}
# the stage named in a check's title, or a model switch's role, and who owns it
STAGE_OWNER = {
    "triage": "Tilly",
    "cleaning plan": "Mop",
    "strategist": "Mop",
    "findings": "Pip",
    "analyst": "Pip",
    "review": "Rex",
    "reviewer": "Rex",
    "writer": "Quill",
    "report": "Quill",
}


def room_origin(col: int, row: int) -> tuple[int, int]:
    return COLS[col], (TOP_Y, BOTTOM_Y)[row]


def spots() -> dict[str, dict[str, list[float]]]:
    """Seat, guest spot, and corridor door for every room (feet-anchored coordinates)."""
    out = {}
    for name, (col, row) in {**ROOMS, "lounge": LOUNGE}.items():
        x, y = room_origin(col, row)
        out[name] = {
            "seat": [x + 190, y + 94],
            "guest": [x + 78, y + 104],
            "guest2": [x + 34, y + 112],  # a second visitor at the same time
            "door": [x + 145, CORRIDOR_Y - 38],
        }
    # where everyone gathers in the lounge at the end
    x, y = room_origin(*LOUNGE)
    out["gather"] = [[x + 40 + 52 * i, y + 120 + (i % 2) * 6] for i in range(len(ROOMS))]
    return out


def _decor(name: str, x: int, y: int) -> str:
    """A little something on the wall and floor of each room."""
    if name == "Tilly":  # a whiteboard checklist
        return (
            f'<rect x="{x + 22}" y="{y + 16}" width="70" height="46" rx="3" fill="#FFFDF8" '
            'stroke="#B9A48C" stroke-width="2"/>'
            + "".join(
                f'<path d="M{x + 30},{y + 28 + i * 11} l4,4 l7,-8" fill="none" stroke="#5E7F33" '
                f'stroke-width="2"/><line x1="{x + 46}" y1="{y + 29 + i * 11}" x2="{x + 84}" '
                f'y2="{y + 29 + i * 11}" stroke="#B9A48C" stroke-width="2"/>'
                for i in range(3)
            )
        )
    if name == "Mop":  # a shelf of cleaning bottles
        bottles = "".join(
            f'<rect x="{x + 28 + i * 18}" y="{y + 30}" width="12" height="22" rx="3" '
            f'fill="{c}"/><rect x="{x + 31 + i * 18}" y="{y + 25}" width="6" height="6" '
            f'fill="#6B5646"/>'
            for i, c in enumerate(("#2F6F73", "#E8B04A", "#8FC1DC", "#B5471B"))
        )
        return (
            f'<rect x="{x + 22}" y="{y + 52}" width="80" height="5" rx="2" fill="#8A6440"/>'
            + bottles
        )
    if name == "Pip":  # a bar chart poster
        bars = "".join(
            f'<rect x="{x + 32 + i * 14}" y="{y + 56 - h}" width="9" height="{h}" fill="{c}"/>'
            for i, (h, c) in enumerate(
                ((14, "#B5471B"), (26, "#2F6F73"), (20, "#C79A1C"), (34, "#7A3E65"))
            )
        )
        return (
            f'<rect x="{x + 22}" y="{y + 14}" width="78" height="50" rx="3" fill="#FFFDF8" '
            f'stroke="#B9A48C" stroke-width="2"/>{bars}'
        )
    if name == "Rex":  # a framed stamp and an in-tray
        return (
            f'<rect x="{x + 22}" y="{y + 14}" width="62" height="46" rx="3" fill="#FFFDF8" '
            f'stroke="#B9A48C" stroke-width="2"/><rect x="{x + 34}" y="{y + 26}" width="38" '
            f'height="22" rx="4" fill="none" stroke="#9E2B25" stroke-width="2.5" '
            f'transform="rotate(-10 {x + 53} {y + 37})"/><text x="{x + 53}" y="{y + 41}" '
            f'text-anchor="middle" font-size="10" fill="#9E2B25" '
            f'transform="rotate(-10 {x + 53} {y + 37})">approved</text>'
        )
    if name == "Quill":  # a bookshelf
        books = "".join(
            f'<rect x="{x + 26 + i * 11}" y="{y + 22 + (i % 3) * 3}" width="8" '
            f'height="{30 - (i % 3) * 3}" fill="{c}"/>'
            for i, c in enumerate(
                ("#B5471B", "#2F6F73", "#C79A1C", "#7A3E65", "#5E7F33", "#4E6E9E", "#B8604F")
            )
        )
        return (
            f'<rect x="{x + 22}" y="{y + 52}" width="84" height="5" rx="2" fill="#8A6440"/>' + books
        )
    return ""


def _room(name: str, label: str, col: int, row: int) -> str:
    x, y = room_origin(col, row)
    door_y = y + ROOM_H - 3 if row == 0 else y - 3
    return (
        f'<rect x="{x}" y="{y}" width="{ROOM_W}" height="{ROOM_H}" rx="10" fill="#FBF6EE" '
        f'stroke="#D9C6A8" stroke-width="3"/>'
        f'<rect x="{x + 3}" y="{y + 3}" width="{ROOM_W - 6}" height="62" rx="8" fill="#F3E9DC"/>'
        f'<rect x="{x + 118}" y="{door_y}" width="54" height="6" fill="#EFE3CF"/>'
        f'<rect x="{x + 200}" y="{y + 12}" width="80" height="20" rx="5" fill="#FFFDF8" '
        f'stroke="#D9C6A8" stroke-width="1.5"/><text x="{x + 240}" y="{y + 26}" '
        f'text-anchor="middle" font-size="11" fill="#6B5646">{label}</text>' + _decor(name, x, y)
    )


def _desk(name: str, col: int, row: int) -> str:
    """The desk front and laptop, drawn over the creature's legs."""
    x, y = room_origin(col, row)
    return (
        f'<g class="desk" data-name="{name}"><rect x="{x + 148}" y="{y + 110}" width="100" '
        f'height="12" rx="3" fill="#A87C52"/><rect x="{x + 152}" y="{y + 122}" width="92" '
        f'height="30" fill="#8A6440"/><rect x="{x + 168}" y="{y + 92}" width="44" height="22" '
        f'rx="3" fill="#6B5646"/><rect class="screen" x="{x + 171}" y="{y + 95}" width="38" '
        f'height="16" rx="2" fill="#6B5646"/><rect x="{x + 222}" y="{y + 100}" width="10" '
        f'height="12" rx="2" fill="#FFFDF8" stroke="#B9A48C" stroke-width="1.5"/></g>'
    )


def _lounge() -> str:
    x, y = room_origin(*LOUNGE)
    drinks = "".join(
        f'<g class="visitor-drink" data-kind="{k}" transform="translate({x + 206},{y + 104})">'
        f"{drink(k)}</g>"
        for k in ("tea", "coffee")
    )
    report = (
        f'<g class="visitor-report" transform="translate({x + 78},{y + 78}) rotate(-8)">'
        '<rect x="-15" y="-19" width="30" height="38" rx="2" fill="#FFFDF8" stroke="#6B5646" '
        'stroke-width="1.5"/><rect x="-15" y="-19" width="30" height="8" rx="2" fill="#B5471B"/>'
        '<path d="M-9,-3 H9 M-9,3 H9 M-9,9 H4" stroke="#6B5646" stroke-width="1.5"/>'
        '<rect x="3" y="-1" width="3" height="10" fill="#2F6F73"/></g>'
    )
    return (
        f'<rect x="{x}" y="{y}" width="{ROOM_W}" height="{ROOM_H}" rx="10" fill="#FBF6EE" '
        f'stroke="#D9C6A8" stroke-width="3"/>'
        f'<rect x="{x + 3}" y="{y + 3}" width="{ROOM_W - 6}" height="62" rx="8" fill="#F3E9DC"/>'
        f'<rect x="{x + 118}" y="{y - 3}" width="54" height="6" fill="#EFE3CF"/>'
        f'<rect x="{x + 22}" y="{y + 14}" width="64" height="40" rx="4" fill="#DCEBF0" '
        f'stroke="#B9A48C" stroke-width="2"/><line x1="{x + 54}" y1="{y + 14}" x2="{x + 54}" '
        f'y2="{y + 54}" stroke="#B9A48C" stroke-width="2"/>'
        f'<rect x="{x + 200}" y="{y + 12}" width="80" height="20" rx="5" fill="#FFFDF8" '
        f'stroke="#D9C6A8" stroke-width="1.5"/><text x="{x + 240}" y="{y + 26}" '
        f'text-anchor="middle" font-size="11" fill="#6B5646">Your corner</text>'
        f'<rect x="{x + 30}" y="{y + 100}" width="96" height="36" rx="12" fill="#B8604F"/>'
        f'<rect x="{x + 36}" y="{y + 86}" width="84" height="24" rx="10" fill="#C9745F"/>'
        f'<rect x="{x + 178}" y="{y + 126}" width="60" height="8" rx="3" fill="#A87C52"/>'
        f'<rect x="{x + 184}" y="{y + 134}" width="6" height="18" fill="#8A6440"/>'
        f'<rect x="{x + 226}" y="{y + 134}" width="6" height="18" fill="#8A6440"/>'
        f'<circle cx="{x + 262}" cy="{y + 118}" r="14" fill="#5E7F33"/>'
        f'<rect x="{x + 254}" y="{y + 128}" width="16" height="18" rx="3" fill="#B5471B"/>'
        + drinks
        + report
    )


def office_svg() -> str:
    labels = {
        "Tilly": "Triage",
        "Mop": "Cleaning",
        "Pip": "Insights",
        "Rex": "Review",
        "Quill": "Reports",
    }
    rooms = "".join(_room(m.name, labels[m.name], *ROOMS[m.name]) for m in TEAM)
    places = spots()
    members = "".join(creature(m, *places[m.name]["seat"], key=f"m-{m.name}") for m in TEAM)
    desks = "".join(_desk(m.name, *ROOMS[m.name]) for m in TEAM)
    return (
        f'<div class="office"><svg id="office-svg" viewBox="0 0 {W} {H}" role="img" '
        f'data-spots="{html.escape(json.dumps(places), quote=True)}" '
        f'aria-label="The Mosaic office: the agents at work on your report">'
        f'<rect x="0" y="0" width="{W}" height="{H}" rx="16" fill="#EFE3CF"/>'
        f"{rooms}{_lounge()}{members}{desks}</svg>"
        f'<div class="office-caption" id="office-caption">The team is getting ready</div></div>'
    )


# ---- live actions from run events ----


def _member(role: str) -> str | None:
    return AGENT_TO_MEMBER.get(role.strip())


def _owner(text: str) -> str | None:
    """Who a check or model switch belongs to, from the stage its title names."""
    lower = text.lower()
    return next((who for stage, who in STAGE_OWNER.items() if stage in lower), None)


def _split(title: str) -> tuple[str, str]:
    """'[table] Rex to Pip: please revise' -> ('table', 'please revise')."""
    part = title[1 : title.index("]")] if title.startswith("[") else ""
    return part, title.split(": ", 1)[-1]


def office_actions(events: list[RunEvent], drink_kind: str = "none") -> list[dict[str, Any]]:
    """Turn the run's events into office actions, in order. Each event maps to actions on
    its own (no look-ahead), so the list only ever grows and the browser can play just the
    new ones."""
    actions: list[dict[str, Any]] = []
    if drink_kind in ("tea", "coffee"):
        actions.append({"type": "deliver", "who": "Tilly", "to": "lounge", "kind": drink_kind})
    working = rejected = None
    for e in events:
        if e.kind == "agent" and e.title.endswith(" is working"):
            who = _member(e.title[: -len(" is working")])
            if who and who != working:
                actions.append({"type": "work", "who": who, "caption": WORKING[who]})
                working = who
        elif e.kind == "message":
            part, subject = _split(e.title)
            sender = _member(e.data.get("sender", ""))
            recipient = _member(e.data.get("recipient", ""))
            team = f"{part.capitalize()} team: " if part else ""
            if e.data.get("sender") == "Cross-Type Synthesizer" or (
                e.data.get("recipient") == "Cross-Type Synthesizer"
            ):
                team = "Linking the types: "
            if not sender:
                continue
            if subject == "approved":
                actions.append({"type": "approve", "who": sender, "say": "Approved!",
                                "caption": f"{team}{sender} approved the findings"})  # fmt: skip
            elif subject == "your summary":
                if part:  # one type's report within a mixed run
                    done = f"{part.capitalize()} done!"
                    actions.append({"type": "say", "who": sender, "say": done,
                                    "caption": f"{team}the {part} report is written"})  # fmt: skip
            elif recipient:
                key = next((k for k in SUBJECT_LINES if subject.startswith(k)), "")
                caption = SUBJECT_CAPTIONS.get(key, "{s} passes work to {r}")
                actions.append({
                    "type": "handoff", "from": sender, "to": recipient,
                    "say": SUBJECT_LINES.get(key, "Over to you!"), "red": key in RED,
                    "caption": team + caption.format(s=sender, r=recipient),
                })  # fmt: skip
            working = None  # after a walk, the next "is working" lights a desk again
        elif e.kind == "guardrail":
            who = _owner(e.title) or working or "Pip"
            say = "Fixing a number" if "fact check" in e.title else "Redoing it"
            last = actions[-1] if actions else {}
            if not (last.get("type") == "reject" and last.get("who") == who):  # one scene
                caption = REDO_CAPTIONS.get(who, f"{who} is fixing something a check caught")
                actions.append({"type": "reject", "who": who, "say": say, "caption": caption})
            rejected = who
        elif e.kind == "fix":
            actions.append({"type": "say", "who": rejected or working or "Pip", "say": "Fixed it!"})
        elif e.kind == "fallback":
            who = _owner(e.title) or working or "Tilly"
            last = actions[-1] if actions else {}
            if not (last.get("type") == "say" and last.get("who") == who
                    and last.get("say") == "Line's busy..."):  # fmt: skip
                actions.append({"type": "say", "who": who, "say": "Line's busy..."})
        elif e.kind == "step" and e.title == "Report ready":
            actions.append({"type": "finale", "who": "Quill", "say": "Your report!"})
        elif e.kind == "error":
            actions.append({"type": "fail", "who": "Rex", "say": "Something went wrong"})
    for i, a in enumerate(actions):
        a["id"] = i
    return actions


def office_state(run_id: str, events: list[RunEvent], drink_kind: str) -> str:
    return json.dumps(
        {"run": run_id, "actions": office_actions(events, drink_kind)}, separators=(",", ":")
    )
