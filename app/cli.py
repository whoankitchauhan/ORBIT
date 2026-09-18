"""Command-line runner for ORBIT.

A fallback for the viva: if the projector, the browser or Streamlit itself
misbehaves, the whole system is still demonstrable from a terminal.

    python -m app.cli --seed
    python -m app.cli "Check complaints for CUST-001 and raise a replacement if eligible"
    python -m app.cli --approve task-ab12cd34ef --by "Dr. Badgujar"
    python -m app.cli --pending
    python -m app.cli --status
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app import runtime
from app.config import settings
from app.graph.state import OrbitState


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 60)}")


def show(state: OrbitState) -> None:
    _rule(f"{state.task_id} - {state.status}")

    print("Plan")
    for step in state.plan:
        print(f"  {step.index + 1}. [{step.status:<8}] {step.agent:<9} {step.instruction[:80]}")

    print("\nTrace")
    for event in state.events:
        print(f"  {event.actor:<11} {event.kind:<8} {event.message[:90]}")

    if state.approval and state.approval.status == "pending":
        _rule("APPROVAL REQUIRED — the workflow is paused")
        print(f"  Action : {state.approval.summary}")
        print(f"  Risk   : {state.approval.risk}")
        print(f"  Because: {state.approval.reason}")
        for key, value in (state.approval.arguments or {}).items():
            print(f"    {key:<14} {value}")
        print(f"\n  Approve with: python -m app.cli --approve {state.task_id}")
        print(f"  Reject with : python -m app.cli --reject  {state.task_id}")
    elif state.final_answer:
        _rule(f"Response · confidence {state.confidence:.0%}")
        print(state.final_answer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orbit", description="Run ORBIT from the terminal.")
    parser.add_argument("objective", nargs="?", help="the objective to run")
    parser.add_argument("--seed", action="store_true", help="load the demo dataset")
    parser.add_argument("--approve", metavar="TASK_ID", help="approve a paused task")
    parser.add_argument("--reject", metavar="TASK_ID", help="reject a paused task")
    parser.add_argument("--by", default="cli-reviewer", help="who is making the decision")
    parser.add_argument("--note", default="", help="a note recorded with the decision")
    parser.add_argument("--pending", action="store_true", help="list tasks awaiting approval")
    parser.add_argument("--status", action="store_true", help="show detected capabilities")
    args = parser.parse_args(argv)

    if args.seed:
        from app.seed import seed_all

        seed_all()
        return 0

    if args.status:
        _rule(f"ORBIT {settings.version}")
        for name, info in settings.capabilities().items():
            marker = "live" if info["mode"] == "live" else "fallback"
            print(f"  {name:<18} {marker:<9} {info['detail']}")
        stats = runtime.system_stats()
        print()
        for key, value in stats.items():
            print(f"  {key:<18} {value}")
        return 0

    if args.pending:
        rows = runtime.pending_approvals()
        if not rows:
            print("Nothing is awaiting approval.")
            return 0
        _rule(f"{len(rows)} task(s) awaiting approval")
        for row in rows:
            print(f"  {row['task_id']}  {row['action']:<28} {row.get('summary', '')[:60]}")
        return 0

    if args.approve or args.reject:
        task_id = args.approve or args.reject
        try:
            runtime.decide_streaming(
                task_id, approved=bool(args.approve), decided_by=args.by, note=args.note
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        state = runtime.get_state(task_id)
        if state:
            show(state)
        return 0

    if not args.objective:
        parser.print_help()
        return 1

    provider = settings.resolved_llm_provider()
    if provider == "simulated":
        print("note: no model API key set — reasoning comes from the offline planner.\n")
    show(runtime.run_task_sync(args.objective))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
