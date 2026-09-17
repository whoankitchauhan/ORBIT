# ORBIT — architecture notes and viva preparation

## 1. One-sentence definition

ORBIT is a stateful multi-agent orchestration platform in which a Supervisor
Agent decomposes an objective, routes each step to a specialist agent with
scoped tool permissions and persistent memory, validates the result, and pauses
for human approval before any consequential action.

## 2. Why this is not a chatbot

A chatbot is `user → LLM → response`. ORBIT is a workflow:

```
user → supervisor → plan → specialists → tools → databases → memory
     → validation → policy check → human approval → external action → response
```

The distinguishing features are **delegation** (the supervisor does no work
itself), **state** (agents communicate through a shared object, not a chat
transcript), **durability** (the run is checkpointed and resumable), and
**control** (an action cannot execute without clearing a policy written in
code).

## 3. The request lifecycle

1. The objective arrives through Streamlit, the CLI or the FastAPI route; all
   three call `app/runtime.py`.
2. `plan_node` asks the Supervisor to decompose it. Retrieved memory and
   similar past tasks are supplied as context.
3. `_normalise_plan` enforces the ordering invariants: research precedes
   analysis, and analysis precedes any action.
4. `supervisor_node` runs once per hop, marks the current step running, and a
   conditional edge routes to the owning specialist.
5. The specialist executes, writes a partial update, and the engine merges it.
   Control returns to the supervisor, which re-decides routing from the new
   state.
6. When the plan is exhausted, or an action has been staged, `validation_node`
   audits the work.
7. If an action is staged, `human_gate_node` evaluates the policy. If approval
   is required it raises `Interrupt`, which checkpoints the run and returns.
8. On approval the engine re-enters the gate node with the decision merged into
   state, falls through, and `action_execute_node` runs the approved call.
9. `finalise_node` writes the response, stores the outcome as an episode in
   long-term memory, and closes the task.

## 4. Design decisions worth defending

**Why a built-in graph engine as well as LangGraph.**
The node functions are written to LangGraph's contract — state in, partial
update out, conditional edges, interrupts — so they are portable. The built-in
engine exists so the pause/resume mechanism is inspectable rather than magic,
and so the project runs on a machine with only Python installed.

**Why the approval policy is code, not a prompt.**
A rule expressed as an instruction to a model can be talked around by a
sufficiently persuasive input. `app/policy.py` is a function. There is no path
from a prompt to a high-risk tool that bypasses it.

**Why the Action Agent has `propose()` and `execute()` as separate methods.**
Because the gate node sits between them in the graph, there is no code path
where an agent decides and acts in one breath. Proposal has no side effects.

**Why the Validation Agent has no tools.**
An auditor that can fetch new evidence tends to quietly fix problems instead of
reporting them, which defeats the purpose of an audit.

**Why confidence matters.**
It is not decoration. Confidence below the threshold escalates to a human, so
an honest low score is what triggers oversight. The Analysis Agent is instructed
to calibrate against evidence strength, not fluency.

**Why tools carry a risk label.**
The policy reads the label. Adding a new external action without marking it
high risk would bypass the gate — so `test_every_external_action_is_high_risk`
fails the build if anyone tries.

## 5. Security model

- Least privilege: each tool declares `allowed_agents`; the registry checks
  before dispatch and returns a *denied* call.
- No free-form SQL. Every query is written in advance; only parameters bind.
  Table names are validated against a whitelist.
- The calculator walks a parsed AST and rejects any non-arithmetic node, so it
  cannot become a code-execution path.
- Document reads resolve and confine paths to the project tree.
- Secrets come from the environment, never source.
- Every tool call, agent run, approval and executed action is written to the
  audit tables.

## 6. Failure handling

| Failure | Behaviour |
|---|---|
| Model API error | Falls back to the offline planner, marks the response |
| Unparsable structured output | `complete_json` returns a typed fallback |
| Tool error or denial | Recorded on the `ToolCall`, surfaced to Validation |
| Persistence failure | The action reports failure; success is never claimed |
| Runaway plan | Hop limit and step limit both cut the run off |
| Refund exceeding the order total | Refused by the tool before execution |
| Process restart mid-approval | Task resumes from the file checkpoint |

## 7. Likely viva questions

**What is an AI agent?** An LLM wrapped in a system that can hold state, choose
tools, observe results and continue until a stopping condition.

**Why multiple agents instead of one?** Specialisation, independent modification,
scoped permissions, traceable failures, and the ability to add a specialist
without redesigning the system. A single agent with every tool is a single
agent with every permission.

**What is orchestration?** Coordinating agents, tools, state and routing so a
complex objective completes as one coherent process.

**What is state?** The shared object holding the objective, the plan, each
agent's output, the risk assessment, the approval and the trace. Nodes return
partial updates, so the diff between checkpoints is exactly what one agent did.

**What is RAG, and how does it differ from memory?** RAG retrieves relevant
context before generation. Memory is the broader capability of retaining
information over time. ORBIT uses the same vector store for both: policies and
handbooks are retrieval; finished tasks written back as episodes are memory.

**What is human-in-the-loop here?** The workflow raises an interrupt, the state
is checkpointed to disk, and execution stops until a person approves or
rejects. On approval the engine re-enters the same node and continues.

**Why PostgreSQL and ChromaDB both?** Different access patterns. Relational
data (tasks, approvals, ledger) needs joins and transactions; semantic recall
needs similarity search over embeddings.

**Why Redis and Celery?** So a long workflow does not block the interface. The
runtime dispatches to Celery when Redis is reachable and to a thread pool
otherwise, with no change at the call site.

**What happens if an agent fails?** The failure is recorded, Validation sees it
as an unresolved issue, confidence drops, and the policy escalates to a human.

**What is the innovation?** Not that it uses several models. It is the
combination — delegation, scoped tool permissions, persistent memory, durable
stateful workflows, enforced human oversight and asynchronous execution — in
one controlled architecture where the oversight cannot be bypassed.

**What is the primary SDG?** SDG 9 — Industry, Innovation and Infrastructure.

## 8. Mapping to the synopsis objectives

| Objective | Where it lives |
|---|---|
| Supervisor-based coordination | `app/agents/supervisor.py`, `app/graph/nodes.py` |
| Tool use and structured intermediate results | `app/tools/`, `app/graph/state.py` |
| Persistent structured state and semantic memory | `app/memory/` |
| Asynchronous and long-running task support | `app/workers/`, `app/runtime.py` |
| Validation and confidence-based escalation | `app/agents/validation_agent.py`, `app/policy.py` |
| Interface showing status, agent activity and results | `frontend/streamlit_app.py` |
