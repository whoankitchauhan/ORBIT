"""The Supervisor Agent — decomposition, routing and final synthesis.

The Supervisor performs no domain work. It decides what should happen, who
should do it, and when the task is finished. Keeping it free of tools is what
prevents it from quietly absorbing the specialists' jobs.
"""

from __future__ import annotations

import time
from typing import Any

from app.agents.base import Agent, AgentResult
from app.config import settings
from app.graph.state import OrbitState, PlanStep
from app.llm import prompts
from app.memory.retrieval import recall_similar_tasks, retrieve_context

VALID_AGENTS = {"research", "analysis", "action"}


def _normalise_plan(plan: list[PlanStep]) -> list[PlanStep]:
    """Apply the ordering rules the Supervisor's prompt states.

    A model can be told "never assign analysis before the research it depends
    on" and still do it, and the offline planner works from keywords that carry
    no notion of dependency. Enforcing the invariants here means they hold
    whichever planner produced the steps:

    * research precedes the first analysis — otherwise there is nothing to
      analyse and the Analysis Agent invents its evidence;
    * analysis precedes the first action — an action with no reasoning behind
      it has nothing to put in the approval card's justification, and the
      reviewer is asked to approve a decision nobody made.
    """
    steps = [s for s in plan if s.instruction.strip()]
    if not steps:
        return steps

    def first_index(agent: str) -> int | None:
        for i, step in enumerate(steps):
            if step.agent == agent:
                return i
        return None

    # Order matters here. Inserting analysis first and research second means
    # the research insert can position itself relative to the analysis step
    # that was just created. Doing it the other way round inserts research
    # after the new analysis and produces analysis → research → action.
    action_at = first_index("action")
    analysis_at = first_index("analysis")
    if action_at is not None and (analysis_at is None or analysis_at > action_at):
        steps.insert(
            action_at,
            PlanStep(index=0, agent="analysis",
                     instruction=f"Decide whether this is justified: {steps[action_at].instruction}"),
        )

    analysis_at = first_index("analysis")
    research_at = first_index("research")
    if analysis_at is not None and (research_at is None or research_at > analysis_at):
        steps.insert(
            analysis_at,
            PlanStep(index=0, agent="research",
                     instruction=f"Gather the evidence needed for: {steps[analysis_at].instruction}"),
        )

    steps = steps[:6]
    for i, step in enumerate(steps):
        step.index = i
    return steps


class SupervisorAgent(Agent):
    name = "supervisor"
    system_prompt = prompts.SUPERVISOR
    description = "Decomposes the objective, routes each step to a specialist, writes the final answer."

    # ------------------------------------------------------------- planning

    def plan(self, state: OrbitState) -> AgentResult:
        """Turn the objective into an ordered list of delegated steps."""
        started = time.perf_counter()

        # Ground planning in memory so the Supervisor can reuse what the
        # system already knows instead of re-deriving it every run.
        recalled = retrieve_context(state.objective, k=settings.retrieval_top_k)
        episodes = recall_similar_tasks(state.objective, k=2)

        memory_block = "\n".join(f"- [{r.kind}] {r.text[:200]}" for r in recalled) or "(none)"
        episode_block = "\n".join(f"- {e.text[:200]}" for e in episodes) or "(none)"

        prompt = (
            f"Objective:\n{state.objective}\n\n"
            f"Relevant knowledge:\n{memory_block}\n\n"
            f"Similar past tasks:\n{episode_block}\n\n"
            "Decompose this objective into ordered steps and assign each to one "
            "specialist agent."
        )

        payload, response = self.llm.complete_json(
            prompt,
            system=self.system_prompt,
            task="plan",
            context={"objective": state.objective, "retrieved": [r.to_dict() for r in recalled]},
            fallback={"steps": [], "rationale": "planner returned unparsable output"},
        )

        raw_steps = payload.get("steps") or []
        plan: list[PlanStep] = []
        for i, raw in enumerate(raw_steps[:6]):
            agent = str(raw.get("agent", "research")).strip().lower()
            if agent not in VALID_AGENTS:
                agent = "research"
            instruction = str(raw.get("instruction", "")).strip()
            if not instruction:
                continue
            plan.append(PlanStep(index=len(plan), agent=agent, instruction=instruction))

        if not plan:
            # A plan is mandatory; degrade to a single research step rather
            # than letting the workflow start with nothing to do.
            plan = [
                PlanStep(index=0, agent="research", instruction=state.objective),
                PlanStep(index=1, agent="analysis", instruction=f"Assess: {state.objective}"),
            ]

        plan = _normalise_plan(plan)
        # A plan of pure research never produces a conclusion, and the
        # Validation Agent will correctly complain that none was recorded.
        if all(s.agent == "research" for s in plan) and len(plan) < 6:
            plan.append(
                PlanStep(index=len(plan), agent="analysis",
                         instruction=f"Draw a conclusion for: {state.objective}")
            )

        latency = int((time.perf_counter() - started) * 1000)
        result = AgentResult(
            agent=self.name,
            output={
                "plan": plan,
                "rationale": payload.get("rationale", ""),
                "retrieved": [r.to_dict() for r in recalled],
            },
            provider=response.provider,
            latency_ms=latency,
        )
        self.record_run(state, -1, state.objective, [s.to_dict() for s in plan],
                        response.provider, latency)
        return result

    # -------------------------------------------------------------- routing

    def route(self, state: OrbitState) -> str:
        """Pick the next node. This is the conditional edge of the graph."""
        if state.hops > settings.max_supervisor_hops:
            return "validation"  # circuit breaker against a looping plan
        step = state.current_step()
        if step is None:
            return "validation"
        return step.agent

    # ------------------------------------------------------- final response

    def synthesise(self, state: OrbitState) -> AgentResult:
        """Write the answer the user actually reads."""
        started = time.perf_counter()
        approval = state.approval.to_dict() if state.approval else None

        prompt = (
            f"Objective: {state.objective}\n\n"
            f"Research: {state.research.get('summary', '(none)')}\n\n"
            f"Analysis: {state.analysis.get('summary', '(none)')}\n\n"
            f"Validation: {state.validation.get('notes', '(none)')} "
            f"(issues: {state.validation.get('issues', [])})\n\n"
            f"Action result: {state.action_result or '(no action taken)'}\n\n"
            f"Human decision: {approval or '(no approval was required)'}\n\n"
            "Write the final response for the user."
        )
        response = self.llm.complete(
            prompt,
            system=prompts.FINAL,
            task="final",
            context={
                "objective": state.objective,
                "research": state.research,
                "analysis": state.analysis,
                "validation": state.validation,
                "action_result": state.action_result,
                "approval": approval,
            },
        )
        latency = int((time.perf_counter() - started) * 1000)
        self.record_run(state, 99, "final synthesis", response.text[:2000],
                        response.provider, latency)
        return AgentResult(
            agent=self.name,
            output={"final_answer": response.text},
            provider=response.provider,
            latency_ms=latency,
        )

    def run(self, state: OrbitState, **kwargs: Any) -> AgentResult:
        return self.plan(state)
