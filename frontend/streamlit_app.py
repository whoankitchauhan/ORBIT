"""ORBIT console — the Streamlit front end.

Run from the project root:
    streamlit run frontend/streamlit_app.py

Five views, each answering a question an examiner or operator will ask:

    Mission       what happens when I give it an objective?
    Approvals     what is waiting on me, and why?
    Tasks         what has this system done?
    Memory        what does it know, and how does it retrieve it?
    Architecture  how is it wired together?
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Make the project importable when Streamlit is launched from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import runtime                                    # noqa: E402
from app.config import settings                            # noqa: E402
from app.graph.state import OrbitState                     # noqa: E402
from app.graph.workflow import graph_diagram, orchestration_backend  # noqa: E402
from app.memory.store import get_store                     # noqa: E402
from app.memory.vector import get_memory                   # noqa: E402
from app.tools.registry import load_all_tools              # noqa: E402
from frontend import theme                                 # noqa: E402

st.set_page_config(
    page_title="ORBIT Console",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject()

EXAMPLES = {
    "Support case with an approval gate": (
        "Check the previous complaints for CUST-001, analyse whether the customer is "
        "eligible for a replacement under our policy, and create a replacement request "
        "if they qualify."
    ),
    "Policy question (no action, completes automatically)": (
        "What is our refund policy for a product delivered 90 days ago, and how much "
        "would a customer get back?"
    ),
    "Comparison across products": (
        "Compare the complaint patterns for the Aurora headphones and the Nimbus "
        "keyboard, and tell me which product line looks more problematic."
    ),
    "Refund above the approval ceiling": (
        "Review complaint CMP-5004 for CUST-003 about the Vega monitor and issue a "
        "refund of Rs 32999 against order ORD-1004 if the dead pixel cluster is covered."
    ),
    "Permission boundary (the research agent may not send mail)": (
        "Send an email to meera.nair@example.com summarising the status of her monitor "
        "complaint."
    ),
}


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------


def bootstrap() -> None:
    """Seed the demo data once per session if the store looks empty."""
    if st.session_state.get("seeded"):
        return
    try:
        store = get_store()
        if not store.query("SELECT id FROM customers LIMIT 1"):
            from app.seed import seed_all

            seed_all(verbose=False)
        elif get_memory().count() == 0:
            from app.seed import seed_knowledge

            seed_knowledge()
    except Exception as exc:
        st.warning(f"Could not prepare demo data: {exc}")
    st.session_state["seeded"] = True


bootstrap()
st.session_state.setdefault("active_task", None)
st.session_state.setdefault("reviewer", "Reviewer")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("#### System status")
    theme.capability_readout(settings.capabilities())

    if settings.resolved_llm_provider() == "simulated":
        st.caption(
            "No model API key is set, so reasoning comes from the built-in offline "
            "planner. Every agent, tool, memory and approval path still runs. Add "
            "a free key (`GEMINI_API_KEY`, `GROQ_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`) "
            "to `.env` for live reasoning."
        )

    st.divider()
    stats = runtime.system_stats()
    st.markdown("#### Totals")
    for label, key in [
        ("Tasks", "tasks"), ("Completed", "completed"), ("Awaiting approval", "awaiting"),
        ("Rejected", "rejected"), ("Agent runs", "agent_runs"),
        ("Tool calls", "tool_calls"), ("Memories", "memories"),
    ]:
        st.markdown(
            f'<div class="cap"><span class="name">{label}</span>'
            f'<span class="val">{stats.get(key, 0)}</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.session_state["reviewer"] = st.text_input(
        "Signed in as", value=st.session_state["reviewer"],
        help="Recorded against every approval decision you make.",
    )


theme.masthead(settings.version)

mission, approvals, tasks, memory, architecture = st.tabs(
    ["Mission", "Approvals", "Tasks", "Memory", "Architecture"]
)


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------


def render_result(state: OrbitState) -> None:
    """Everything worth reading about a finished or paused task."""
    left, right = st.columns([3, 2], gap="large")

    with left:
        if state.final_answer:
            st.markdown("#### Response")
            st.markdown(state.final_answer)
        elif state.status == "awaiting_approval":
            st.markdown("#### Waiting on you")
            st.caption("Open the Approvals tab to release or stop this action.")

        if state.research.get("findings"):
            with st.expander(f"Evidence · {len(state.research['findings'])} finding(s)"):
                for finding in state.research["findings"]:
                    st.markdown(f"- {finding}")
                sources = ", ".join(state.research.get("sources", [])) or "none"
                st.caption(f"Sources: {sources}")

        if state.analysis.get("conclusion"):
            with st.expander("Analysis"):
                st.markdown(state.analysis["conclusion"])
                for observation in state.analysis.get("observations", []):
                    st.markdown(f"- {observation}")

        if state.validation:
            issues = state.validation.get("issues", [])
            with st.expander(f"Validation · {'passed' if not issues else f'{len(issues)} issue(s)'}"):
                if issues:
                    for issue in issues:
                        st.markdown(f"- {issue}")
                else:
                    st.markdown("All automated checks passed.")

        if state.tool_calls:
            with st.expander(f"Tool calls · {len(state.tool_calls)}"):
                st.dataframe(
                    [
                        {
                            "tool": c.name, "agent": c.agent, "risk": c.risk,
                            "status": c.status, "ms": c.duration_ms,
                            "detail": c.error or str(c.result)[:120],
                        }
                        for c in state.tool_calls
                    ],
                    width="stretch", hide_index=True,
                )

    with right:
        st.markdown("#### Plan")
        theme.plan_sequence([s.to_dict() for s in state.plan])
        st.markdown("#### Trace")
        st.markdown(theme.trace([e.to_dict() for e in state.events]), unsafe_allow_html=True)


with mission:
    st.markdown("### Submit an objective")
    st.caption(
        "The Supervisor decomposes it, routes each step to a specialist, and stops for "
        "you before anything consequential happens."
    )

    choice = st.selectbox(
        "Start from an example", ["Write my own"] + list(EXAMPLES),
        help="These exercise different paths: automatic completion, the approval gate, "
             "the confidence floor and the permission model.",
    )
    default = EXAMPLES.get(choice, "")
    objective = st.text_area("Objective", value=default, height=110,
                             placeholder="e.g. Check CUST-002's complaint history and raise a replacement if it qualifies")

    go, _ = st.columns([1, 4])
    with go:
        launch = st.button("Run workflow", type="primary", width="stretch")

    if launch:
        if not objective.strip():
            st.error("Enter an objective first.")
        else:
            plan_slot = st.empty()
            trace_slot = st.empty()
            progress_slot = st.empty()

            def on_step(node: str, state: OrbitState) -> None:
                progress_slot.markdown(
                    f'<div style="font-size:0.82rem;color:#5C6B82">Executing '
                    f'<span class="mono" style="color:#0E7C86">{theme.esc(node)}</span> · '
                    f'{len(state.events)} events</div>',
                    unsafe_allow_html=True,
                )
                with plan_slot.container():
                    theme.plan_sequence([s.to_dict() for s in state.plan])
                trace_slot.markdown(
                    theme.trace([e.to_dict() for e in state.events]), unsafe_allow_html=True
                )

            with st.spinner("Orchestrating…"):
                try:
                    final = runtime.run_task_streaming(
                        objective, user_id=st.session_state["reviewer"], on_step=on_step
                    )
                    st.session_state["active_task"] = final.task_id
                except Exception as exc:
                    st.error(f"The workflow could not start: {exc}")

            # The live view is replaced by the settled view rendered below.
            plan_slot.empty()
            trace_slot.empty()
            progress_slot.empty()

    # Rendered on every run, not only the one that launched the workflow.
    # Streamlit reruns the whole script when a button is clicked, so anything
    # guarded by `if launch:` would vanish the instant the reviewer pressed
    # Approve — and the handler behind that button would never fire.
    active = st.session_state.get("active_task")
    if active:
        state = runtime.get_state(active)
        if state is None:
            st.caption("The workflow checkpoint for this task is no longer available.")
        else:
            st.divider()
            st.markdown(
                f'<span class="mono" style="font-size:0.8rem;color:#5C6B82">'
                f'{theme.esc(state.task_id)}</span> &nbsp; {theme.chip(state.status)}',
                unsafe_allow_html=True,
            )

            if state.status == "awaiting_approval" and state.approval:
                theme.approval_gate(state.approval.to_dict(), state.objective)
                note = st.text_input(
                    "Decision note", key="mission_note",
                    placeholder="Why are you approving or rejecting this?",
                )
                approve_col, reject_col = st.columns(2)
                if approve_col.button("Approve and continue", type="primary",
                                      width="stretch", key="inline_approve"):
                    with st.spinner("Resuming the workflow…"):
                        try:
                            runtime.decide_streaming(
                                state.task_id, True, st.session_state["reviewer"], note
                            )
                        except Exception as exc:
                            st.error(f"Could not resume: {exc}")
                    st.rerun()
                if reject_col.button("Reject", width="stretch", key="inline_reject"):
                    with st.spinner("Closing out…"):
                        try:
                            runtime.decide_streaming(
                                state.task_id, False, st.session_state["reviewer"], note
                            )
                        except Exception as exc:
                            st.error(f"Could not resume: {exc}")
                    st.rerun()
            elif state.status == "completed":
                st.success(f"Completed · confidence {state.confidence:.0%}")
            elif state.status == "rejected":
                st.warning("Closed without acting — the proposed action was rejected.")
            elif state.status == "failed":
                st.error("The workflow failed. The trace below shows where.")

            render_result(state)


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------

with approvals:
    st.markdown("### Awaiting a human decision")
    pending = runtime.pending_approvals()

    if not pending:
        st.caption(
            "Nothing is waiting. Run the support-case example on the Mission tab to "
            "put a workflow in front of you."
        )
    else:
        st.caption(
            f"{len(pending)} workflow(s) frozen mid-execution. Each is checkpointed to "
            "disk, so they survive a restart of this app."
        )
        for row in pending:
            import json

            try:
                arguments = json.loads(row.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}

            theme.approval_gate(
                {
                    "summary": row.get("summary", ""),
                    "reason": row.get("reason", ""),
                    "arguments": arguments,
                },
                row.get("objective", ""),
            )
            note = st.text_input(
                "Decision note", key=f"note_{row['id']}",
                placeholder="Why are you approving or rejecting this?",
            )
            approve_col, reject_col, id_col = st.columns([1, 1, 3])
            if approve_col.button("Approve", type="primary", key=f"ok_{row['id']}",
                                  width="stretch"):
                with st.spinner("Resuming the workflow…"):
                    try:
                        runtime.decide_streaming(
                            row["task_id"], True, st.session_state["reviewer"], note
                        )
                        st.success("Approved — the action has been executed.")
                    except Exception as exc:
                        st.error(f"Could not resume: {exc}")
                time.sleep(0.6)
                st.rerun()
            if reject_col.button("Reject", key=f"no_{row['id']}", width="stretch"):
                with st.spinner("Closing the workflow…"):
                    try:
                        runtime.decide_streaming(
                            row["task_id"], False, st.session_state["reviewer"], note
                        )
                        st.info("Rejected — nothing was executed.")
                    except Exception as exc:
                        st.error(f"Could not resume: {exc}")
                time.sleep(0.6)
                st.rerun()
            id_col.markdown(
                f'<div style="padding-top:0.4rem"><span class="mono" '
                f'style="font-size:0.75rem;color:#5C6B82">{theme.esc(row["task_id"])}</span></div>',
                unsafe_allow_html=True,
            )
            st.divider()

    with st.expander("Decision history"):
        history = get_store().list_approvals(30)
        if history:
            st.dataframe(
                [
                    {
                        "action": h["action"], "status": h["status"],
                        "risk": h["risk"], "decided by": h["decided_by"] or "—",
                        "note": (h["note"] or "—")[:60], "task": h["task_id"],
                    }
                    for h in history
                ],
                width="stretch", hide_index=True,
            )
        else:
            st.caption("No decisions recorded yet.")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

with tasks:
    st.markdown("### Task history")
    rows = runtime.list_tasks(60)

    if not rows:
        st.caption("No tasks yet.")
    else:
        stats = runtime.system_stats()
        theme.metric_strip([
            ("Total", stats["tasks"]), ("Completed", stats["completed"]),
            ("Awaiting", stats["awaiting"]), ("Rejected", stats["rejected"]),
            ("Actions executed", stats["actions"]),
        ])
        st.write("")

        labels = {
            f"{r['objective'][:72]}{'…' if len(r['objective']) > 72 else ''}  ·  {r['status']}": r["id"]
            for r in rows
        }
        selected = st.selectbox("Open a task", list(labels), label_visibility="collapsed")
        task_id = labels[selected]

        state = runtime.get_state(task_id)
        record = get_store().get_task(task_id)
        if record:
            st.markdown(
                f'<span class="mono" style="font-size:0.8rem;color:#5C6B82">'
                f'{theme.esc(task_id)}</span> &nbsp; {theme.chip(record["status"])} '
                f'&nbsp; <span style="font-size:0.8rem;color:#5C6B82">confidence '
                f'{float(record.get("confidence") or 0):.0%}</span>',
                unsafe_allow_html=True,
            )
        if state:
            render_result(state)
        else:
            st.caption("The workflow checkpoint for this task is no longer on disk.")

        with st.expander("Agent runs"):
            runs = get_store().list_agent_runs(task_id)
            if runs:
                st.dataframe(
                    [
                        {
                            "agent": r["agent_name"], "step": r["step_index"],
                            "status": r["status"], "provider": r["provider"],
                            "ms": r["latency_ms"],
                        }
                        for r in runs
                    ],
                    width="stretch", hide_index=True,
                )
            else:
                st.caption("No agent runs recorded.")

        with st.expander("Executed actions (audit ledger)"):
            ledger = get_store().list_actions(30)
            if ledger:
                st.dataframe(
                    [
                        {"action": a["action"], "reference": a["outcome"],
                         "task": a["task_id"], "payload": (a["payload"] or "")[:90]}
                        for a in ledger
                    ],
                    width="stretch", hide_index=True,
                )
            else:
                st.caption("Nothing has been executed yet.")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

with memory:
    store_memory = get_memory()
    st.markdown("### Semantic memory")
    st.caption(
        f"{store_memory.count()} record(s) in the {store_memory.backend} backend. "
        "Policies, handbooks and complaint records are retrieved by meaning, and every "
        "finished task is written back as an episode."
    )

    search_col, k_col = st.columns([4, 1])
    query = search_col.text_input(
        "Search memory", placeholder="e.g. when is a customer eligible for a replacement?",
        label_visibility="collapsed",
    )
    top_k = k_col.number_input("Results", 1, 12, 5, label_visibility="collapsed")

    if query:
        results = store_memory.search(query, k=int(top_k))
        if not results:
            st.caption("Nothing matched.")
        for record in results:
            st.markdown(
                f'<div class="panel"><h4>{theme.esc(record.source)} '
                f'{theme.chip(record.kind if record.kind in theme.STATUS_COLOURS else "pending")} '
                f'<span class="mono" style="font-size:0.75rem;color:#5C6B82;font-weight:400">'
                f'relevance {record.score:.3f}</span></h4>'
                f'<p>{theme.esc(record.text[:600])}</p></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### Add to the knowledge base")
    with st.form("ingest", clear_on_submit=True):
        text = st.text_area(
            "Text", height=130,
            placeholder="Paste a policy, handbook section or product note. It will be chunked and embedded.",
        )
        source_col, kind_col = st.columns(2)
        source = source_col.text_input("Source label", value="manual-upload")
        kind = kind_col.selectbox("Kind", ["policy", "document", "record", "episode"])
        upload = st.file_uploader("…or upload a .txt / .md file", type=["txt", "md"])

        if st.form_submit_button("Store in memory", type="primary"):
            content = text
            label = source
            if upload is not None:
                content = upload.read().decode("utf-8", errors="replace")
                label = upload.name
            if not content.strip():
                st.error("Nothing to store.")
            else:
                chunks = store_memory.chunk_and_add(content, source=label, kind=kind)
                st.success(f"Stored {chunks} chunk(s) from {label}.")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

with architecture:
    from app.graph.nodes import AGENTS

    st.markdown("### How ORBIT is wired")
    st.caption(
        f"Orchestration backend: {orchestration_backend()} · structured store: "
        f"{get_store().backend} · semantic memory: {get_memory().backend}"
    )

    st.markdown("#### Workflow graph")
    st.code(graph_diagram(), language="text")
    st.caption(
        "Specialists return to the Supervisor after every step, so routing is decided "
        "from current state on each hop rather than fixed in advance. The gate node "
        "raises an interrupt, which checkpoints the run and hands control to a person."
    )

    st.markdown("#### Agents and their permissions")
    for name, agent in AGENTS.items():
        card = agent.card()
        rows = theme.tool_rows(card["tools"]) or (
            '<div class="tool-desc" style="padding:0.4rem 0">'
            "No tools — this agent reasons over what other agents produced.</div>"
        )
        st.markdown(
            f'<div class="panel"><h4 style="color:'
            f'{theme.ACTOR_COLOURS.get(name, theme.INK)}">{theme.esc(name)} agent</h4>'
            f'<p style="color:#5C6B82">{theme.esc(card["description"])}</p>'
            f'<div style="margin-top:0.6rem">{rows}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Escalation policy")
    st.markdown(
        f"""<div class="panel"><p>A workflow stops for a human when any of these hold.
        The rules are Python, not prompt text, so an agent cannot argue its way past them.</p>
        <ul style="font-size:0.86rem;color:#16233A;line-height:1.7">
          <li>The proposed tool is classified <span class="mono">high</span> risk.</li>
          <li>Confidence falls below {settings.approval_confidence_threshold:.0%}.</li>
          <li>The Validation Agent left unresolved issues.</li>
          <li>A monetary amount exceeds the configured ceiling.</li>
          <li>The action sends something to an external recipient.</li>
          <li>Required arguments for the call are missing.</li>
        </ul></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("#### Full tool registry")
    registry = load_all_tools()
    st.markdown(
        f'<div class="panel">{theme.tool_rows([{"name": t.name, "risk": t.risk, "description": t.description} for t in registry.all()])}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Permissions are enforced at call time: if an agent requests a tool outside its "
        "allowance the registry returns a denied call rather than executing it."
    )
