"""Visual language for the ORBIT console.

Design notes
------------
The reference point is an instrument panel, not a chat app: the operator is
watching a process they may need to intervene in, so state and provenance are
always on screen.

*Light, cool paper* rather than the usual dark console. This gets demonstrated
on a lecture-theatre projector, where dark backgrounds wash out to mud and
low-contrast text disappears from the back row.

*Colour carries one meaning each.* Teal marks live infrastructure, slate marks
fallbacks, green completion, rose failure. Amber is spent on exactly one thing
— a workflow waiting for a human — so an approval is unmistakable at a glance
from across a room. Nothing else on the page is allowed to use it.

*IBM Plex Sans and Plex Mono.* Plex was drawn for engineering interfaces and
brings the right register. The mono face is functional rather than decorative:
identifiers like ORD-1001 and ORD-l00l must be told apart, and telemetry
figures need to align in columns.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

INK = "#16233A"
MUTED = "#5C6B82"
LINE = "#D3DAE5"
PAPER = "#EEF1F5"
SURFACE = "#FFFFFF"
TEAL = "#0E7C86"
AMBER = "#B26A12"
AMBER_WASH = "#FDF3E2"
GREEN = "#2E6F4E"
ROSE = "#A33A3A"

STATUS_COLOURS = {
    "pending": (MUTED, "#E8ECF2"),
    "queued": (MUTED, "#E8ECF2"),
    "running": (TEAL, "#DDF0F1"),
    "awaiting_approval": (AMBER, AMBER_WASH),
    "completed": (GREEN, "#DFEFE6"),
    "rejected": (ROSE, "#F7E4E4"),
    "failed": (ROSE, "#F7E4E4"),
    "done": (GREEN, "#DFEFE6"),
    "skipped": (MUTED, "#E8ECF2"),
    "ok": (GREEN, "#DFEFE6"),
    "error": (ROSE, "#F7E4E4"),
    "denied": (ROSE, "#F7E4E4"),
    # Risk levels. "high" shares the amber of the approval gate deliberately:
    # a high-risk tool is precisely the thing that will stop for a human, so
    # the colour means the same thing in both places.
    "low": (MUTED, "#E8ECF2"),
    "medium": ("#3B4E86", "#E4E9F6"),
    "high": (AMBER, AMBER_WASH),
}

ACTOR_COLOURS = {
    "supervisor": "#3B4E86",
    "research": TEAL,
    "analysis": "#6B4E9C",
    "action": "#9C5A2E",
    "validation": "#2E6F4E",
    "human": AMBER,
    "system": MUTED,
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, .stApp, [class*="css"] {{
    font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}

h1, h2, h3, h4 {{ color: {INK}; font-weight: 600; letter-spacing: -0.015em; }}

code, .mono {{ font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }}

section[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {LINE};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}

/* ---------- masthead ---------- */
.orbit-head {{
    display: flex; align-items: baseline; gap: 0.9rem;
    border-bottom: 2px solid {INK}; padding-bottom: 0.7rem; margin-bottom: 0.4rem;
}}
.orbit-head .mark {{
    font-size: 1.85rem; font-weight: 700; letter-spacing: 0.02em; color: {INK};
}}
.orbit-head .expand {{ font-size: 0.95rem; color: {MUTED}; font-weight: 400; }}
.orbit-sub {{ color: {MUTED}; font-size: 0.86rem; margin-bottom: 1.4rem; }}

/* ---------- chips ---------- */
.chip {{
    display: inline-block; padding: 0.12rem 0.55rem; border-radius: 3px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
    font-family: 'IBM Plex Mono', monospace;
}}

/* ---------- capability readout ---------- */
.cap {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.42rem 0; border-bottom: 1px dotted {LINE}; font-size: 0.82rem;
}}
.cap:last-child {{ border-bottom: none; }}
.cap .name {{ color: {INK}; font-weight: 500; }}
.cap .val {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; text-align: right; }}
.cap .live {{ color: {TEAL}; }}
.cap .fallback {{ color: {MUTED}; }}

/* ---------- plan sequence ---------- */
.seq {{ border-left: 2px solid {LINE}; margin: 0.3rem 0 1rem 0.5rem; padding-left: 0; }}
.seq-step {{ position: relative; padding: 0.5rem 0 0.5rem 1.6rem; }}
.seq-step::before {{
    content: attr(data-n); position: absolute; left: -0.72rem; top: 0.55rem;
    width: 1.35rem; height: 1.35rem; border-radius: 50%;
    background: {SURFACE}; border: 2px solid {LINE}; color: {MUTED};
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 600;
    display: flex; align-items: center; justify-content: center;
}}
.seq-step.done::before {{ border-color: {GREEN}; color: {GREEN}; }}
.seq-step.running::before {{ border-color: {TEAL}; color: {TEAL}; }}
.seq-step.failed::before {{ border-color: {ROSE}; color: {ROSE}; }}
.seq-agent {{ font-weight: 600; font-size: 0.83rem; }}
.seq-instruction {{ color: {MUTED}; font-size: 0.83rem; line-height: 1.45; margin-top: 0.1rem; }}

/* ---------- trace ---------- */
.trace {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 4px;
    padding: 0.55rem 0.8rem; max-height: 420px; overflow-y: auto;
}}
.trace-line {{
    display: grid; grid-template-columns: 4.6rem 5.6rem 1fr; gap: 0.7rem;
    padding: 0.28rem 0; border-bottom: 1px solid #F1F4F8; font-size: 0.8rem;
    align-items: baseline;
}}
.trace-line:last-child {{ border-bottom: none; }}
.trace-ts {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {MUTED}; }}
.trace-actor {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600; }}
.trace-msg {{ color: {INK}; line-height: 1.4; }}

/* ---------- the approval gate: the one place amber appears ---------- */
.gate {{
    background: {AMBER_WASH}; border: 1px solid #E8C486;
    border-left: 5px solid {AMBER}; border-radius: 4px;
    padding: 1.1rem 1.3rem; margin: 0.5rem 0 1rem 0;
}}
.gate-label {{
    font-size: 0.74rem; font-weight: 600; color: {AMBER};
    font-family: 'IBM Plex Mono', monospace; margin-bottom: 0.45rem;
}}
.gate-action {{
    font-size: 1.22rem; font-weight: 600; color: {INK};
    line-height: 1.35; margin-bottom: 0.6rem;
}}
.gate-why {{ font-size: 0.85rem; color: #6B4A18; line-height: 1.5; }}
.gate-why strong {{ color: {AMBER}; font-weight: 600; }}
.gate-args {{
    margin-top: 0.75rem; padding-top: 0.65rem; border-top: 1px solid #E8C486;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: #6B4A18;
}}
.gate-args div {{ padding: 0.1rem 0; }}
.gate-args .k {{ display: inline-block; min-width: 8.5rem; color: {AMBER}; }}

/* ---------- panels ---------- */
.panel {{
    background: {SURFACE}; border: 1px solid {LINE};
    border-radius: 4px; padding: 0.95rem 1.15rem; margin-bottom: 0.85rem;
}}
.panel h4 {{ margin: 0 0 0.55rem 0; font-size: 0.93rem; }}
.panel p {{ font-size: 0.86rem; line-height: 1.55; color: {INK}; margin: 0.2rem 0; }}

/* ---------- metric strip ---------- */
.strip {{ display: flex; gap: 0; border: 1px solid {LINE}; border-radius: 4px; background: {SURFACE}; }}
.strip .cell {{ flex: 1; padding: 0.7rem 0.9rem; border-right: 1px solid {LINE}; }}
.strip .cell:last-child {{ border-right: none; }}
.strip .n {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem; font-weight: 600; color: {INK}; }}
.strip .l {{ font-size: 0.72rem; color: {MUTED}; margin-top: 0.1rem; }}

/* ---------- tool table ---------- */
.tool-row {{
    display: grid; grid-template-columns: 13rem 5rem 1fr; gap: 0.8rem;
    padding: 0.42rem 0; border-bottom: 1px solid #F1F4F8; font-size: 0.82rem;
    align-items: baseline;
}}
.tool-name {{ font-family: 'IBM Plex Mono', monospace; font-weight: 500; color: {INK}; }}
.tool-desc {{ color: {MUTED}; line-height: 1.4; }}

.stButton button {{ border-radius: 3px; font-weight: 500; }}
div[data-testid="stExpander"] details {{
    border: 1px solid {LINE}; border-radius: 4px; background: {SURFACE};
}}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value: Any) -> str:
    return html.escape(str(value))


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def chip(status: str) -> str:
    fg, bg = STATUS_COLOURS.get(status, (MUTED, "#E8ECF2"))
    label = status.replace("_", " ")
    return f'<span class="chip" style="color:{fg};background:{bg}">{esc(label)}</span>'


def masthead(version: str) -> None:
    st.markdown(
        f"""<div class="orbit-head">
              <span class="mark">ORBIT</span>
              <span class="expand">Orchestrated Reasoning &amp; Behavioral Intelligence Technology</span>
            </div>
            <div class="orbit-sub">
              Multi-agent orchestration with persistent memory, tool use and human-in-the-loop
              escalation &nbsp;·&nbsp; <span class="mono">v{esc(version)}</span>
            </div>""",
        unsafe_allow_html=True,
    )


def capability_readout(capabilities: dict[str, dict[str, str]]) -> None:
    rows = "".join(
        f'<div class="cap"><span class="name">{esc(name)}</span>'
        f'<span class="val {info["mode"]}">{esc(info["detail"])}</span></div>'
        for name, info in capabilities.items()
    )
    st.markdown(rows, unsafe_allow_html=True)


def metric_strip(cells: list[tuple[str, Any]]) -> None:
    body = "".join(
        f'<div class="cell"><div class="n">{esc(value)}</div><div class="l">{esc(label)}</div></div>'
        for label, value in cells
    )
    st.markdown(f'<div class="strip">{body}</div>', unsafe_allow_html=True)


def plan_sequence(plan: list[dict[str, Any]]) -> None:
    """The delegated steps, numbered because they genuinely are a sequence."""
    if not plan:
        st.markdown('<p style="color:#5C6B82;font-size:0.85rem">No plan yet.</p>',
                    unsafe_allow_html=True)
        return
    steps = "".join(
        f'<div class="seq-step {esc(step["status"])}" data-n="{i + 1}">'
        f'<div class="seq-agent" style="color:{ACTOR_COLOURS.get(step["agent"], INK)}">'
        f'{esc(step["agent"])} agent {chip(step["status"])}</div>'
        f'<div class="seq-instruction">{esc(step["instruction"])}</div></div>'
        for i, step in enumerate(plan)
    )
    st.markdown(f'<div class="seq">{steps}</div>', unsafe_allow_html=True)


def trace(events: list[dict[str, Any]], limit: int = 200) -> str:
    """Execution trace. Returned as HTML so it can be written into a placeholder."""
    if not events:
        return '<div class="trace"><div class="trace-line"><span class="trace-msg" style="color:#5C6B82">Waiting for the first node…</span></div></div>'

    import datetime as _dt

    lines = []
    for event in events[-limit:]:
        stamp = _dt.datetime.fromtimestamp(event["ts"]).strftime("%H:%M:%S")
        colour = ACTOR_COLOURS.get(event["actor"], MUTED)
        lines.append(
            f'<div class="trace-line">'
            f'<span class="trace-ts">{stamp}</span>'
            f'<span class="trace-actor" style="color:{colour}">{esc(event["actor"])}</span>'
            f'<span class="trace-msg">{esc(event["message"])}</span></div>'
        )
    return f'<div class="trace">{"".join(lines)}</div>'


def approval_gate(approval: dict[str, Any], objective: str = "") -> None:
    """The one amber element: a workflow frozen, waiting for a person."""
    reasons = "".join(
        f"<div>· {esc(r.strip())}</div>" for r in approval.get("reason", "").split(";") if r.strip()
    )
    arguments = "".join(
        f'<div><span class="k">{esc(k)}</span>{esc(v)}</div>'
        for k, v in (approval.get("arguments") or {}).items()
    )
    context = (
        f'<div style="font-size:0.8rem;color:#6B4A18;margin-bottom:0.5rem">'
        f"Objective: {esc(objective[:160])}</div>" if objective else ""
    )
    st.markdown(
        f"""<div class="gate">
              <div class="gate-label">WORKFLOW PAUSED · APPROVAL REQUIRED</div>
              {context}
              <div class="gate-action">{esc(approval.get("summary", "Action pending"))}</div>
              <div class="gate-why"><strong>Why you are being asked</strong>{reasons}</div>
              {f'<div class="gate-args">{arguments}</div>' if arguments else ""}
            </div>""",
        unsafe_allow_html=True,
    )


def panel(title: str, body_html: str) -> None:
    st.markdown(f'<div class="panel"><h4>{esc(title)}</h4>{body_html}</div>',
                unsafe_allow_html=True)


def tool_rows(tools: list[dict[str, Any]]) -> str:
    return "".join(
        f'<div class="tool-row"><span class="tool-name">{esc(t["name"])}</span>'
        f'{chip(t["risk"])}'
        f'<span class="tool-desc">{esc(t["description"])}</span></div>'
        for t in tools
    )
