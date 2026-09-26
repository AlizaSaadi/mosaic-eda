"""The Mosaic team: SVG creatures, one per agent, shared by the welcome screen and the office.

Each creature is drawn around (0, 0) (its feet at y=38) so it can be placed anywhere with
a translate. The outer <g> is moved by the office script (walking); the inner
<g class="crt"> bobs, blinks, hops, and stresses out with CSS.
"""

from __future__ import annotations

from dataclasses import dataclass

INK = "#2B1D14"
CHEEK = "#F3B9A0"
SWEAT = "#8FC1DC"


@dataclass(frozen=True)
class Member:
    name: str
    role: str
    agent: str  # the CrewAI role name this creature plays
    body: str
    feet: str
    blurb: str


TEAM = [
    Member("Tilly", "Triage lead", "Dataset Triage Lead", "#B5471B", "#8C3512",
           "Reads your data first and decides what matters."),
    Member("Mop", "Cleaning strategist", "Cleaning Strategist", "#2F6F73", "#1F4C4F",
           "Plans the cleaning, using only allowed operations."),
    Member("Pip", "Insight analyst", "Insight Analyst", "#C79A1C", "#7A5A00",
           "Finds the patterns. Every number gets fact-checked."),
    Member("Rex", "Senior reviewer", "Senior Reviewer", "#7A3E65", "#55294A",
           "Checks the reasoning and sends work back if needed."),
    Member("Quill", "Report writer", "Report Writer", "#5E7F33", "#3F5722",
           "Writes the summary you'll read at the end."),
]  # fmt: skip

# Special analysts play Pip's part in the office
AGENT_TO_MEMBER = {m.agent: m.name for m in TEAM} | {
    "Video Synthesizer": "Pip",
    "Cross-Type Synthesizer": "Pip",
}

_BODIES = {
    "Tilly": "M0,-40 C24,-40 38,-22 38,0 C38,25 22,38 0,38 C-22,38 -38,25 -38,0 C-38,-22 -24,-40 0,-40Z",
    "Mop": "M0,-36 C26,-36 36,-18 36,2 C36,26 20,38 0,38 C-20,38 -36,26 -36,2 C-36,-18 -26,-36 0,-36Z",
    "Pip": "M0,-34 C22,-34 34,-18 34,4 C34,26 20,38 0,38 C-20,38 -34,26 -34,4 C-34,-18 -22,-34 0,-34Z",
    "Rex": "M0,-42 C23,-42 37,-24 37,0 C37,25 21,38 0,38 C-21,38 -37,25 -37,0 C-37,-24 -23,-42 0,-42Z",
    "Quill": "M0,-36 C24,-36 36,-19 36,2 C36,26 21,38 0,38 C-21,38 -36,26 -36,2 C-36,-19 -24,-36 0,-36Z",
}  # fmt: skip

_EYES = {
    "Tilly": (-12, -10),
    "Mop": (-12, -8),
    "Pip": (-11, -6),
    "Rex": (-12, -12),
}


def _eyes(name: str) -> str:
    if name == "Quill":  # happy closed eyes
        return (
            f'<g class="eyes"><path d="M-19,-8 Q-12,-15 -5,-8" fill="none" stroke="{INK}" '
            f'stroke-width="2.5" stroke-linecap="round"/><path d="M5,-8 Q12,-15 19,-8" '
            f'fill="none" stroke="{INK}" stroke-width="2.5" stroke-linecap="round"/></g>'
        )
    x, y = _EYES[name]
    return (
        f'<g class="eyes"><circle cx="{x}" cy="{y}" r="8" fill="#fff"/>'
        f'<circle cx="{-x}" cy="{y}" r="8" fill="#fff"/>'
        f'<circle cx="{x + 1.5}" cy="{y + 1}" r="3.5" fill="{INK}"/>'
        f'<circle cx="{-x + 1.5}" cy="{y + 1}" r="3.5" fill="{INK}"/></g>'
    )


def _props(name: str) -> tuple[str, str]:
    """(drawn behind the body, drawn in front of it)."""
    s = f'stroke="{INK}"'
    if name == "Tilly":
        front = (
            f'<rect x="24" y="0" width="20" height="26" rx="3" fill="#F3E9DC" {s} '
            'stroke-width="1.5"/><rect x="29" y="-3" width="10" height="5" rx="1.5" '
            'fill="#6B5646"/><path d="M28,9 H40 M28,14 H40 M28,19 H36" stroke="#6B5646" '
            f'stroke-width="1.5"/><circle cx="-12" cy="-10" r="10.5" fill="none" {s} '
            f'stroke-width="2"/><circle cx="12" cy="-10" r="10.5" fill="none" {s} '
            f'stroke-width="2"/><line x1="-1.5" y1="-10" x2="1.5" y2="-10" {s} stroke-width="2"/>'
        )
        return "", front
    if name == "Mop":
        back = (
            '<line x1="-44" y1="-26" x2="-30" y2="30" stroke="#8A6440" stroke-width="3.5" '
            'stroke-linecap="round"/><path d="M-36,26 L-22,24 L-18,40 L-40,42 Z" '
            'fill="#E8B04A" stroke="#8A6440" stroke-width="1.5"/>'
        )
        front = '<path d="M-20,-30 Q0,-46 20,-30 Q10,-36 0,-35 Q-10,-36 -20,-30Z" fill="#E8B04A"/>'
        return back, front
    if name == "Pip":
        back = (
            f'<line x1="-2" y1="-34" x2="-4" y2="-46" {s} stroke-width="2" '
            'stroke-linecap="round"/><circle cx="-4" cy="-47" r="3.5" fill="#B5471B"/>'
        )
        front = (
            '<line x1="44" y1="8" x2="54" y2="20" stroke="#6B5646" stroke-width="4" '
            'stroke-linecap="round"/><circle cx="37" cy="-1" r="11" fill="#DCEBF0" '
            f'fill-opacity=".55" {s} stroke-width="2.5"/>'
        )
        return back, front
    if name == "Rex":
        front = (
            f'<path d="M-11,-26 L-3,-24 M11,-26 L3,-24" {s} stroke-width="2.2" '
            'stroke-linecap="round"/><path d="M0,20 L-12,13 L-12,27 Z M0,20 L12,13 L12,27 Z" '
            'fill="#9E2B25"/><circle cx="0" cy="20" r="3" fill="#6E1D19"/>'
            '<line x1="30" y1="0" x2="46" y2="-18" stroke="#9E2B25" stroke-width="4.5" '
            f'stroke-linecap="round"/><line x1="46" y1="-18" x2="49" y2="-21" {s} '
            'stroke-width="2.5" stroke-linecap="round"/>'
        )
        return "", front
    back = (  # Quill: a draft in hand and a pencil behind the head
        '<rect x="-48" y="-4" width="22" height="28" rx="2" fill="#FFFDF8" stroke="#6B5646" '
        'stroke-width="1.5" transform="rotate(-8 -37 10)"/><path d="M-44,4 L-31,2" '
        'stroke="#B5471B" stroke-width="1.5"/><path d="M-43,10 L-30,8 M-42,16 L-33,15" '
        'stroke="#6B5646" stroke-width="1.5"/>'
    )
    front = (
        '<g transform="rotate(-28 14 -40)"><rect x="-4" y="-44" width="30" height="7" rx="1.5" '
        'fill="#E8B04A"/><path d="M26,-44 L34,-40.5 L26,-37 Z" fill="#F3E9DC"/>'
        f'<path d="M31.5,-41.6 L34,-40.5 L31.5,-39.4 Z" fill="{INK}"/><rect x="-8" y="-44" '
        'width="5" height="7" rx="1" fill="#B8604F"/></g>'
    )
    return back, front


def _mouth(name: str) -> str:
    smile = "M-6,6 L6,6" if name == "Rex" else "M-6,10 Q0,16 6,10"
    return (
        f'<path class="smile" d="{smile}" fill="none" stroke="{INK}" stroke-width="2" '
        f'stroke-linecap="round"/><path class="worry" d="M-6,14 Q0,9 6,14" fill="none" '
        f'stroke="{INK}" stroke-width="2" stroke-linecap="round"/>'
    )


def creature(member: Member, x: float = 0, y: float = 0, *, key: str = "") -> str:
    """One creature as an SVG group, with a paper to carry, a speech bubble, and a sweat drop
    that the office script and CSS show when needed."""
    back, front = _props(member.name)
    cheeks = (
        f'<ellipse cx="-21" cy="7" rx="5" ry="3" fill="{CHEEK}" opacity=".8"/>'
        f'<ellipse cx="21" cy="7" rx="5" ry="3" fill="{CHEEK}" opacity=".8"/>'
    )
    feet = (
        f'<ellipse cx="-14" cy="38" rx="9" ry="4.5" fill="{member.feet}"/>'
        f'<ellipse cx="14" cy="38" rx="9" ry="4.5" fill="{member.feet}"/>'
    )
    paper = (
        '<g class="paper"><rect x="-10" y="16" width="20" height="24" rx="2" fill="#FFFDF8" '
        'stroke="#6B5646" stroke-width="1.5"/><path d="M-6,23 H6 M-6,28 H6 M-6,33 H2" '
        'stroke="#6B5646" stroke-width="1.5"/><path class="redmark" d="M-8,18 L8,38 M8,18 '
        'L-8,38" stroke="#9E2B25" stroke-width="2.5" stroke-linecap="round"/></g>'
    )
    sweat = (
        f'<path class="sweat" d="M30,-36 C33,-30 35,-27 32,-24 C29,-22 26,-25 27,-28 Z" '
        f'fill="{SWEAT}"/>'
    )
    bubble = (
        '<g class="bubble"><rect x="-46" y="-86" width="92" height="26" rx="8" fill="#FFFDF8" '
        f'stroke="{INK}" stroke-width="1.2"/><path d="M-6,-60.5 L0,-52 L6,-60.5" fill="#FFFDF8" '
        f'stroke="{INK}" stroke-width="1.2"/><text x="0" y="-68.5" text-anchor="middle" '
        f'font-size="11" fill="{INK}" class="say"></text></g>'
    )
    ident = f' id="{key}"' if key else ""
    return (
        f'<g class="member" data-name="{member.name}"{ident} transform="translate({x},{y})">'
        f'<g class="crt" data-name="{member.name}">{feet}{back}'
        f'<path d="{_BODIES[member.name]}" fill="{member.body}"/>{_eyes(member.name)}{cheeks}'
        f"{_mouth(member.name)}{front}{sweat}{paper}</g>{bubble}</g>"
    )


def team_lineup() -> str:
    """The welcome screen's row of wobbling creatures with their names and roles."""
    xs = [80, 220, 360, 500, 640]
    members = "".join(
        f'<g class="intro" style="animation-delay:{0.15 + i * 0.18:.2f}s">{creature(m, x, 118)}</g>'
        for i, (m, x) in enumerate(zip(TEAM, xs, strict=True))
    )
    cards = "".join(
        f'<div class="card"><div class="n">{m.name}</div><div class="r">{m.role}</div>'
        f'<div class="b">{m.blurb}</div></div>'
        for m in TEAM
    )
    return (
        '<div class="lineup"><svg viewBox="0 0 720 176" role="img" aria-label="The Mosaic '
        'team: Tilly, Mop, Pip, Rex, and Quill"><ellipse cx="360" cy="160" rx="340" ry="6" '
        f'fill="#E4D5BF" opacity=".6"/>{members}</svg><div class="cards">{cards}</div></div>'
    )


def upload_icon(done: bool = False) -> str:
    """A folder that a sheet keeps dropping into; turns into a check when a file is in."""
    state = " done" if done else ""
    return (
        f'<div class="upload-icon{state}" aria-hidden="true"><svg viewBox="0 0 120 96">'
        '<g class="sheet"><rect x="44" y="4" width="32" height="40" rx="3" fill="#FFFDF8" '
        'stroke="#6B5646" stroke-width="2"/><path d="M50,14 H70 M50,21 H70 M50,28 H64" '
        'stroke="#B5471B" stroke-width="2" stroke-linecap="round"/></g>'
        '<path d="M16,36 H48 L56,44 H104 V84 Q104,88 100,88 H20 Q16,88 16,84 Z" '
        'fill="#E8B04A"/><path d="M12,50 H108 L100,88 H20 Z" fill="#C79A1C"/>'
        '<g class="check"><circle cx="60" cy="66" r="15" fill="#5E7F33"/>'
        '<path d="M52,66 L58,72 L69,60" fill="none" stroke="#fff" stroke-width="3.5" '
        'stroke-linecap="round" stroke-linejoin="round"/></g></svg></div>'
    )


def drink(kind: str) -> str:
    """The visitor's tea or coffee, for the office corner (empty for 'none')."""
    if kind == "tea":
        cup = (
            '<ellipse cx="0" cy="22" rx="26" ry="5" fill="#E4D5BF"/>'
            '<path d="M-16,-2 H16 L12,20 H-12 Z" fill="#FFFDF8" stroke="#6B5646" '
            'stroke-width="1.5"/><path d="M16,3 Q26,5 16,14" fill="none" stroke="#6B5646" '
            'stroke-width="2"/><ellipse cx="0" cy="-1" rx="15" ry="3" fill="#B8604F"/>'
            '<path d="M6,-2 L10,-16" stroke="#6B5646" stroke-width="1"/>'
            '<rect x="7" y="-22" width="7" height="7" rx="1" fill="#5E7F33"/>'
        )
    elif kind == "coffee":
        cup = (
            '<path d="M-14,-4 H14 V16 Q14,22 8,22 H-8 Q-14,22 -14,16 Z" fill="#B5471B"/>'
            '<path d="M14,1 Q24,3 14,13" fill="none" stroke="#B5471B" stroke-width="3"/>'
            '<ellipse cx="0" cy="-4" rx="14" ry="3" fill="#F3E9DC"/>'
            '<ellipse cx="0" cy="-4" rx="9" ry="1.8" fill="#C79A1C"/>'
        )
    else:
        return ""
    steam = (
        '<g class="steam"><path d="M-6,-10 q-4,-6 0,-12 q4,-6 0,-12" fill="none" '
        'stroke="#B9A48C" stroke-width="2" stroke-linecap="round"/><path d="M4,-10 q-4,-6 0,-12 '
        'q4,-6 0,-12" fill="none" stroke="#B9A48C" stroke-width="2" stroke-linecap="round"/></g>'
    )
    return f'<g class="drink">{steam}{cup}</g>'
