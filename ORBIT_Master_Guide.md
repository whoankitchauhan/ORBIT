# ORBIT Master Guide

**Orchestrated Reasoning & Behavioral Intelligence Technology**  
Publication-grade technical and operational reference  
Version 1.0.0

> This guide describes the repository as it exists. Where the project exposes an optional enterprise dependency, the guide names the production role of that dependency and separately identifies the built-in fallback. This distinction is important during evaluation: installing LangGraph, PostgreSQL, ChromaDB, Redis, or Celery does not by itself change every runtime path.

---

## Contents

1. Executive Summary and Foundational Vision
2. End-to-End Processing Pipeline
3. Codebase and File-by-File Breakdown
4. Setup, Deployment, and Operational Runbook
5. Technical Defense and Viva Voce Manual
6. Design Limits and Production Hardening Checklist
7. Appendix: API and Data Contracts

---

# 1. Executive Summary and Foundational Vision

## 1.1 What ORBIT is

ORBIT is a stateful, multi-agent workflow platform. It accepts an objective, decomposes it into specialist work, retrieves evidence, performs analysis, proposes a consequential action, validates the result, and asks a human to approve risky work before execution.

A useful workplace analogy is a structured corporate office:

- The **Supervisor** is the project manager. It breaks a request into an ordered plan, assigns work, checks progress, and writes the final brief.
- The **Research Agent** is the records and investigations team. It reads policies, documents, customer records, and prior cases.
- The **Analysis Agent** is the analyst. It calculates, compares, and converts evidence into a conclusion.
- The **Action Agent** is the operations desk. It prepares and, only after authorization, performs one business action.
- The **Validation Agent** is an independent reviewer. It checks that the conclusion is supported, complete, and policy-safe.
- The **human approval gate** is the manager's signature. The system cannot treat a proposed refund, email, replacement, or record change as executed until a reviewer decides.
- The **shared state** is the case file. It contains the objective, plan, evidence, calculations, risk, approval request, trace, errors, and final answer.
- The **tool registry** is the office access-control desk. It decides which department may use which capability.

The current repository implements this design with a built-in graph engine. LangGraph is detected as an optional capability and the node contract is intentionally compatible with LangGraph-style nodes, but `app/graph/workflow.py` currently compiles the repository's own `app/graph/engine.py` graph.

## 1.2 The core problem

A traditional single-turn chatbot follows this simplified path:

```text
User -> one model call -> text response
```

That shape is acceptable for drafting or casual questions. Enterprise work is different:

```text
Request -> evidence -> reasoning -> policy check -> approval -> side effect -> audit -> response
```

A monolithic model with every tool and every permission tends to fail in several ways:

| Enterprise requirement | Why a monolithic model is weak | ORBIT response |
|---|---|---|
| Multi-step reasoning | A long prompt mixes planning, retrieval, calculation, and action. Earlier assumptions can be forgotten or overwritten. | A plan and typed shared state separate stages and preserve intermediate results. |
| Permission scoping | If one agent can call every tool, a prompt injection or routing mistake can reach a write operation. | Tools declare allowed agents and risk; the registry enforces access before dispatch. |
| Error recovery | One failed completion often returns a plausible but incomplete answer. | Tool failures, validation issues, confidence, and graph errors are recorded and routed into review. |
| Human accountability | Text saying "I think this is safe" is not an authorization record. | Policy code creates an approval request with action, arguments, reason, reviewer, and note. |
| Durability | An in-memory chat turn disappears when a process restarts. | Checkpoints are written atomically to `data/checkpoints`; structured task records are also stored. |
| Auditability | A final paragraph does not show which data or tools produced it. | Events, agent runs, tool calls, approvals, workflow snapshots, and action ledger rows are retained. |
| Context control | Passing the entire transcript to every role increases cost and distracts the model. | Each worker receives a compact context digest and only the fields it needs. |

## 1.3 Architectural advantage

ORBIT decomposes work across a Supervisor and specialist workers. The decomposition provides four practical controls:

1. **Context control.** Research returns evidence; Analysis receives the evidence rather than an ever-growing conversation. The state has explicit fields for research, analysis, validation, and action results.
2. **Specialization.** Each role has a narrow purpose and output contract. The Supervisor coordinates but has no tools. The Validation Agent has no tools so that it reports defects rather than silently repairing its own audit.
3. **Permission boundaries.** Research can read; Analysis can calculate; Action can invoke business operations. The registry rejects a call before the function is looked up when the agent is not allowed to use that tool.
4. **Controlled side effects.** `ActionAgent.propose()` is separate from `ActionAgent.execute()`. Validation and policy assessment occur between them. High-risk tools stop at the human gate.

This also reduces hallucination impact. ORBIT does not assume that model confidence is truth: it asks for evidence, runs deterministic checks, lowers confidence for unresolved issues, and escalates uncertain or consequential cases.

## 1.4 Core component rationale

| Component | Practical role | Current repository reality |
|---|---|---|
| **LangGraph** | A state-machine model with nodes, conditional routing, cycles, and interrupts. Useful for making long-running agent workflows explicit. | Optional capability detection only. The built-in engine executes the current workflow. |
| **FastAPI** | Async-friendly HTTP gateway for task submission, status, approval, memory ingestion, health, and integration with a future React client. | Implemented in `app/api/routes.py`; the handlers are synchronous Python functions but are safe to expose through Uvicorn. |
| **PostgreSQL** | Durable relational storage for tasks, audit records, approvals, tool calls, business data, and reporting. | Selected when `DATABASE_URL` and a PostgreSQL driver are available; otherwise SQLite is used. Graph resume still uses file checkpoints. |
| **ChromaDB** | Similarity retrieval over embeddings for policies, handbooks, complaints, and completed task episodes. | Selected when the package is installed; otherwise an embedded JSON index uses hashed-token vectors plus lexical overlap. |
| **Redis** | Low-latency broker and result backend for distributed work. | Reachability is tested with `ping()`. |
| **Celery** | Worker process that runs long workflows and document ingestion outside the request/UI process. | Enabled only when `USE_CELERY=true`, Celery is installed, and Redis is reachable. |
| **Streamlit** | Demonstration console and human approval dashboard. | The primary UI with Mission, Approvals, Tasks, Memory, and Architecture tabs. |
| **React** | A production-grade alternative frontend for a separate API client. | Not present in this repository. FastAPI is the integration boundary a React application could use. |

## 1.5 Architectural invariants

The following rules are more important than any particular model vendor:

- Research precedes analysis, and analysis precedes action. The Supervisor normalizes plans to preserve this order.
- Action proposal has no side effect.
- Every external action is high risk in the current registry and therefore requires approval.
- The graph has a hop limit and an execution step limit.
- Every node receives complete state and returns a partial update.
- List fields such as events and tool calls append during merge; scalar fields replace their prior value.
- Offline results are labelled `simulated` and are not represented as hosted-model output.
- A rejected approval finalizes without executing the pending action.

---

# 2. End-to-End Processing Pipeline

## 2.1 Pipeline overview

```text
Client (Streamlit, CLI, API)
        |
        v
runtime.submit_task / run_task_*
        |
        v
OrbitState(task_id, objective, user_id)
        |
        v
plan -> supervisor -> specialist -> supervisor ...
                         | research / analysis / action proposal
                         v
                    validation
                         |
             pending action? -- no --> finalise
                         |
                        yes
                         v
                 human approval gate
                    |           |
                 approve       reject
                    |           |
                 execute     finalise
                    |
                    v
                 supervisor -> validation/finalise
```

The normal synchronous path is useful for the CLI and tests. The API normally calls `submit_task(..., background=true)`, which returns a task ID immediately and uses Celery when enabled or a four-worker thread pool otherwise. Streamlit deliberately runs inline with callbacks so the trace can be rendered as nodes complete.

## 2.2 Step 1: ingestion

A request can enter through:

- Streamlit Mission tab, which calls `runtime.run_task_streaming()`.
- CLI, which calls the runtime service layer.
- `POST /tasks`, whose `TaskRequest` accepts an objective from 3 to 4000 characters, a `user_id`, and a `background` flag.

`runtime.submit_task()` strips the objective, rejects an empty value, creates a `task-*` identifier, constructs `OrbitState`, inserts an initial task row, and records a queued progress snapshot. It then dispatches to Celery, the thread pool, or inline execution according to capability and caller choice.

Example request:

```http
POST /tasks
Content-Type: application/json

{"objective":"Check complaints for CUST-001 and raise a replacement if eligible","user_id":"api-user","background":true}
```

Example immediate response:

```json
{"task_id":"task-xxxxxxxxxx","status":"accepted"}
```

The task is accepted, not necessarily completed. Poll `GET /tasks/{task_id}` for state and progress.

## 2.3 Step 2: state initialization

`OrbitState` is a dataclass rather than an unstructured chat transcript. Its important fields are:

- Identity: `task_id`, `objective`, `user_id`, timestamps.
- Control: `status`, `cursor`, `hops`, `risk`, `confidence`.
- Work: `plan`, `research`, `analysis`, `validation`, `action_result`.
- Memory: `retrieved`.
- Approval: `pending_action`, `approval`.
- Audit: `events`, `tool_calls`, `errors`.
- Result: `final_answer`.

`plan_node()` marks the task as running, logs receipt, persists the initial task, and asks `SupervisorAgent.plan()` to produce `PlanStep` records. A plan step has an index, agent name, instruction, status, output key, and timing fields. The supervisor retrieves relevant memory and similar task context before planning.

## 2.4 Step 3: Supervisor routing

The graph first moves from `plan` to `supervisor`. Each supervisor visit increments `state.hops`, locates the next pending or running plan step, marks it running, and records the instruction. `route_from_supervisor()` returns one of `research`, `analysis`, `action`, or `validation`.

The return to the Supervisor after Research, Analysis, and Execute is intentional. It means routing is reevaluated from current state instead of being a fixed hard-coded chain. The circuit breaker marks pending steps skipped once `MAX_SUPERVISOR_HOPS` is exceeded and routes to validation.

## 2.5 Step 4: worker execution and tools

### Research

`ResearchAgent.run()` receives the current state and step instruction. It extracts useful identifiers such as customer IDs, order IDs, and email addresses, retrieves semantic context, selects read-only tools, invokes at most four selected calls for a step, and returns findings and a summary. Typical tools include:

- `knowledge_search`, `policy_lookup`, and `recall_past_tasks`.
- `lookup_customer`, `customer_orders`, and `customer_complaints`.
- `read_document` and other document functions.

The registry checks agent permission, required arguments, risk metadata, and then calls the function. Each invocation becomes a `ToolCall` and is persisted by the base agent/runtime path.

### Analysis

`AnalysisAgent.run()` works on research output. It can call `summarise_numbers`, `calculate`, and `compare_options`. The numeric support path is deterministic and produces statistics. The analysis output includes observations, a conclusion, a verdict, statistics, and calibrated confidence.

### Action proposal

`ActionAgent.propose()` chooses one action and constructs arguments. It does not dispatch the external operation. `action_propose_node()` calls `assess_action()` and stores `pending_action`, risk, and, if needed, an `ApprovalRequest`. In the current implementation the external action tools are high risk, so the normal consequential path pauses.

### Action execution

After approval, `action_execute_node()` calls `ActionAgent.execute()` with the stored proposal. Current external API functions are demonstration adapters: `_dispatch()` writes to `action_ledger` and does not contact a live CRM or mail provider. The functions still enforce business invariants such as refund not exceeding the order total and replacement ownership checks.

## 2.6 Step 5: state mutation and checkpointing

A node returns only the fields it changed. `CompiledGraph._apply()` merges the update into the in-memory `OrbitState`. Lists such as `events`, `tool_calls`, `errors`, and `retrieved` are appended; other fields are replaced.

After each successful node, the engine writes a `Checkpoint(thread_id, next_node, state, step)`. The default workflow uses `FileCheckpointer`, which writes JSON to `data/checkpoints` through a temporary file and atomic replace. This gives restart durability for paused work.

The structured store separately records tasks, agent runs, tools, approvals, workflow-state snapshots, and action ledger rows. The current graph's authoritative resume location is the file checkpoint, not a PostgreSQL checkpoint table. `save_workflow_state()` provides an additional SQL snapshot at finalization and is suitable groundwork for a future database-backed checkpointer.

## 2.7 Step 6: human-in-the-loop interruption

The exact pause/resume sequence is:

1. Action proposal is created with a tool and arguments.
2. Validation runs before the approval request is shown.
3. `route_after_validation()` sends a pending action to `gate`.
4. `human_gate_node()` sees a pending approval and logs the reason.
5. It saves the approval and task, then raises `Interrupt`.
6. `CompiledGraph.invoke()` catches `Interrupt` before applying a node update, writes a checkpoint whose `next_node` is `gate`, and returns `RunResult(interrupted=True, ...)`.
7. Runtime sets task status to `awaiting_approval`.
8. Streamlit reads pending approvals, or an API client reads `GET /approvals`.
9. A reviewer posts `POST /approvals/{task_id}/approve` or `/reject` with `decided_by` and `note`.
10. `runtime.decide_streaming()` or `_decide()` loads the checkpoint, updates the approval status, and calls `workflow.resume()`.
11. Resume reloads serialized state, merges the decision, and re-enters `gate`.
12. Approved state routes to `execute`; rejected state marks pending steps skipped and routes to `finalise`.

The gate is durable across a Streamlit restart because the checkpoint is on disk. The reviewer name is recorded but is not authenticated in this version.

## 2.8 Step 7: validation and synthesis

`ValidationAgent.run()` performs deterministic checks for evidence, attribution, coverage, conclusion support, failed tools, objective coverage, and incomplete steps. It may combine these with model-produced issues. Unresolved issues lower confidence.

`finalise_node()` asks `SupervisorAgent.synthesise()` to produce the final Markdown response, selects `completed` or `rejected`, stores the result as an episodic memory record, writes final structured state, and logs completion. If the action was rejected, the final response explains that no action was executed.

## 2.9 Exact graph topology

ASCII execution topology:

```text
                         +-------------+
                         |    plan     |
                         +------+------+ 
                                |
                         +------v------+
                         |  supervisor |
                         +--+---+---+--+
                            |   |   |
                 +----------+   |   +----------+
                 |              |              |
          +------v-----+ +-----v------+ +-----v-----+
          |  research  | |  analysis  | |   action  |
          |  read-only | | calculations| | propose   |
          +------+-----+ +-----+------+ +-----+-----+
                 |              |              |
                 +--------------+--------------+
                                |
                         +------v------+
                         |  supervisor |
                         +------+------+
                                |
                         +------v------+
                         | validation  |
                         +------+------+
                                |
                         +------v------+
                         | pending ?   |
                         +---+------+---+
                         no  |      | yes
                             |      |
                      +------v--+ +--v-------+
                      | finalise | |   gate   |
                      +---------+ +--+----+--+
                                    |    |
                              reject|    |approve
                                    |    |
                             +------v+  +v---------+
                             |finalise|  | execute |
                             +--------+  +----+----+
                                               |
                                               v
                                          supervisor
```

State transition diagram:

```text
pending -> running -> awaiting_approval -> running -> completed
                         |                         |
                         +-------------------------+
                         |
                         +-> rejected -> finalise

running -> failed
```

Sequence diagram:

```text
Client        Runtime       Graph        Supervisor       Worker       Store/Checkpoint     Reviewer
  |              |             |              |              |                 |                |
  | POST task    |             |              |              |                 |                |
  |------------->|             |              |              |                 |                |
  |              | create state              |              |                 |                |
  |              |-------------------------->|              |                 |                |
  |              |             | plan         |              |                 |                |
  |              |             |------------->|              |                 |                |
  |              |             | route        |              |                 |                |
  |              |             |--------------------------------------------->| checkpoint     |
  |              |             |                            run tools          |                |
  |              |             |-------------------------------->|             |                |
  |              |             |             update + validate  |             |                |
  |              |             |<----------------------------------------------------------|
  |              |             | interrupt at gate             |             |                |
  |<-------------| awaiting approval                         |             |                |
  | GET approvals|             |              |              |                 |                |
  |------------------------------------------------------------------------------------------->|
  |              |             | resume(decision)             |             |                |
  | POST approve |             |<-----------------------------------------------------------|
  |------------->|             |              |              |                 |                |
  |              |             | execute if approved          |             |                |
  |              |             |-------------------------------->|             |                |
  |              |             | finalise                     |             |                |
  |<-------------| completed   |              |              |                 |                |
```

---

# 3. Exhaustive Codebase and File-by-File Breakdown

## 3.1 Directory map

```text
ORBIT/
|-- .env.example
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- requirements-minimal.txt
|-- run.bat
|-- run.sh
|-- .streamlit/config.toml
|-- app/
|   |-- __init__.py
|   |-- cli.py
|   |-- config.py
|   |-- policy.py
|   |-- runtime.py
|   |-- seed.py
|   |-- agents/
|   |   |-- __init__.py
|   |   |-- action_agent.py
|   |   |-- analysis_agent.py
|   |   |-- base.py
|   |   |-- research_agent.py
|   |   |-- supervisor.py
|   |   |-- validation_agent.py
|   |-- api/
|   |   |-- __init__.py
|   |   |-- routes.py
|   |-- graph/
|   |   |-- __init__.py
|   |   |-- engine.py
|   |   |-- nodes.py
|   |   |-- state.py
|   |   |-- workflow.py
|   |-- llm/
|   |   |-- __init__.py
|   |   |-- offline.py
|   |   |-- prompts.py
|   |   |-- provider.py
|   |-- memory/
|   |   |-- __init__.py
|   |   |-- retrieval.py
|   |   |-- store.py
|   |   |-- vector.py
|   |-- tools/
|   |   |-- __init__.py
|   |   |-- calculator.py
|   |   |-- database.py
|   |   |-- documents.py
|   |   |-- external_api.py
|   |   |-- registry.py
|   |   |-- search.py
|   |-- workers/
|       |-- __init__.py
|       |-- celery_app.py
|       |-- tasks.py
|-- data/.gitkeep
|-- docs/ARCHITECTURE.md
|-- frontend/
|   |-- __init__.py
|   |-- streamlit_app.py
|   |-- theme.py
|-- tests/
    |-- __init__.py
    |-- test_graph.py
    |-- test_tools.py
    |-- test_workflow.py
```

Generated runtime data is normally placed under `data/`: `orbit.db`, `chroma/`, `checkpoints/`, and embedded vector JSON. These are ignored by Git.

## 3.2 Root and configuration files

| File | Single responsibility | Inputs and outputs | Plain-language logic and interfaces |
|---|---|---|---|
| `.env.example` | Document optional environment settings. | Environment variables in; runtime settings out indirectly. | Lists model, database, vector, queue, policy, API, and data-directory keys. |
| `.gitignore` | Keep secrets, databases, checkpoints, indexes, caches, and virtual environments out of version control. | Git paths in; ignore rules out. | Prevents local state and credentials from becoming source artifacts. |
| `README.md` | Quick-start and project overview. | Reader in; commands, architecture summary, limitations out. | The shortest orientation document and demonstration script. |
| `requirements.txt` | Full optional dependency set. | pip in; Python packages out. | Includes Streamlit, model SDKs, LangGraph, LangChain core, psycopg, ChromaDB, Celery, Redis, FastAPI, Uvicorn, Pydantic, and pytest. |
| `requirements-minimal.txt` | Minimal demo dependency set. | pip in; Streamlit and pytest out. | The application supplies fallbacks for the remaining components. |
| `run.bat` | Windows launcher. | Shell invocation in; seeded Streamlit process out. | Changes to repository directory, runs `python -m app.seed`, starts Streamlit. |
| `run.sh` | Unix-like launcher. | Shell invocation and optional Streamlit args in; seeded Streamlit process out. | Same behavior as `run.bat`, with `set -e`. |
| `.streamlit/config.toml` | Streamlit theme/server configuration. | Streamlit in; visual/server settings out. | Keeps the console presentation consistent. |
| `frontend/.streamlit/config.toml` | Frontend-local Streamlit configuration. | Streamlit in; frontend server/theme settings out. | Allows the frontend package to carry the same local presentation settings when launched from its directory. |
| `data/.gitkeep` | Preserve the generated-data directory in Git. | Git in; empty directory placeholder out. | Runtime data is created beside it. |

## 3.3 Application services and seed data

| File | Responsibility | Inputs and outputs | Adjacent interfaces |
|---|---|---|---|
| `app/__init__.py` | Package marker. | Python import in; package out. | Enables `app.*` imports. |
| `app/config.py` | Load settings and detect capabilities. | `.env`, environment, installed modules, Redis reachability in; `settings`, `DATA_DIR`, capability map out. | Used by every service that chooses live versus fallback behavior. |
| `app/policy.py` | Enforce escalation rules. | State and action proposal in; `PolicyDecision` and plain-language reason out. | Reads tool registry metadata and policy settings; called before human gate. |
| `app/runtime.py` | Common service layer for all frontends and workers. | Objective, user, background mode, approval decision in; task IDs, states, progress, and final states out. | Calls graph workflow, Store, Celery task wrappers, and thread pool. |
| `app/seed.py` | Populate demo business and knowledge data. | Built-in customers, orders, complaints, policies, and handbook text in; relational rows and vector chunks out. | Uses Store and VectorMemory; safe to rerun for demo data. |
| `app/cli.py` | Terminal interface. | Commands, objective, approval options in; status, traces, and final result out. | Calls `runtime`; useful when Streamlit or browser access is inconvenient. |

## 3.4 Agents

| File | Responsibility | Inputs and outputs | Adjacent interfaces |
|---|---|---|---|
| `app/agents/__init__.py` | Re-export agent classes and base types. | Imports in; public agent package out. | Used by callers that want a single import surface. |
| `app/agents/base.py` | Shared agent infrastructure. | State, tool name, arguments, agent identity in; `AgentResult`, audited tool calls, and persisted runs out. | Connects agents to LLM provider, registry, and Store. |
| `app/agents/supervisor.py` | Plan, route, and synthesize. | Objective, retrieved context, state in; normalized `PlanStep` list and final Markdown answer out. | Called by plan, supervisor, and finalise nodes; has no tools. |
| `app/agents/research_agent.py` | Gather grounded evidence. | State and instruction in; research findings, summary, retrieved context, and tool audit records out. | Uses read-only registry tools and vector retrieval. |
| `app/agents/analysis_agent.py` | Calculate and judge. | Research state and instruction in; conclusion, observations, verdict, statistics, confidence out. | Uses calculator tools and LLM provider. |
| `app/agents/action_agent.py` | Separate proposal from execution. | State and instruction/proposal in; proposal or action result out. | Uses external API tools only after policy and gate. |
| `app/agents/validation_agent.py` | Audit evidence and consistency. | Full state in; passed flag, issues, checks, confidence out. | Called after action proposal or completed plan; intentionally has no tools. |

## 3.5 API and graph

| File | Responsibility | Inputs and outputs | Adjacent interfaces |
|---|---|---|---|
| `app/api/__init__.py` | API package marker. | Python import in; package out. | Supports `app.api.routes`. |
| `app/api/routes.py` | FastAPI HTTP contract. | JSON requests and query parameters in; JSON health, tasks, approvals, tools, memory, stats, and graph responses out. | Calls runtime, vector memory, graph diagram, and registry. |
| `app/graph/__init__.py` | Graph package marker. | Python import in; package out. | Supports graph modules. |
| `app/graph/state.py` | Typed workflow data model. | State fields and JSON dictionaries in; `OrbitState`, plan steps, events, tool calls, approvals, serialization out. | Every node and checkpoint depends on its serialization contract. |
| `app/graph/engine.py` | Execute nodes, conditional edges, interrupts, and checkpoints. | Graph definitions, state, thread ID, callbacks in; `RunResult`, updated state, or `GraphError` out. | Used by workflow assembly and runtime; independent of external LangGraph. |
| `app/graph/nodes.py` | Implement graph behavior. | `OrbitState` in; partial dictionaries out; `Interrupt` from gate. | Bridges agents, policy, Store, retrieval, and engine. |
| `app/graph/workflow.py` | Assemble the concrete ORBIT topology. | Settings and node functions in; compiled file-checkpointed graph, Mermaid diagram, backend label out. | Runtime calls `get_workflow`; frontend displays `graph_diagram()`. |

## 3.6 LLM layer

| File | Responsibility | Inputs and outputs | Adjacent interfaces |
|---|---|---|---|
| `app/llm/__init__.py` | Re-export LLM client functions. | Imports in; `LLMClient` and `get_llm()` out. | Simplifies agent imports. |
| `app/llm/provider.py` | Uniform hosted/offline model access. | Prompt, system contract, task, context in; `LLMResponse` or parsed JSON plus response out. | Agents call this instead of vendor SDKs directly. Falls back on provider failure and JSON parse failure when a fallback is supplied. |
| `app/llm/offline.py` | Rule-based offline planner and responder. | Task type, prompt, context in; deterministic structured text out. | Used when no provider is configured or a provider call fails. |
| `app/llm/prompts.py` | Role and output contracts. | Agent context in; prompt/system text out. | Keeps supervisor, research, analysis, validation, final, and risk prompts consistent. |

## 3.7 Memory and retrieval

| File | Responsibility | Inputs and outputs | Adjacent interfaces |
|---|---|---|---|
| `app/memory/__init__.py` | Re-export structured and vector memory APIs. | Imports in; public memory package out. | Used by agents, API, seed, and workers. |
| `app/memory/store.py` | Relational data-access layer. | SQL parameters and domain objects in; task rows, agent runs, tools, approvals, states, action ledger, and stats out. | Uses PostgreSQL when configured and SQLite otherwise. All SQL values are bound; database tools add table whitelists. |
| `app/memory/vector.py` | Embedding index abstraction. | Text, source, kind, query, k in; stored chunks and ranked results out. | Uses ChromaDB if installed, otherwise JSON persistence with hashed-token vectors and lexical overlap. |
| `app/memory/retrieval.py` | RAG and episodic-memory helpers. | Query, state, source records in; context prompts, similar tasks, and memory IDs out. | Research and finalization use it; vector store is the underlying index. |

## 3.8 Tools and registry

| File | Responsibility | Inputs and outputs | Adjacent interfaces |
|---|---|---|---|
| `app/tools/__init__.py` | Re-export registry types and loader. | Imports in; public tool package out. | Used by agents and API. |
| `app/tools/registry.py` | Metadata, permissions, validation, and dispatch. | Tool definitions, agent identity, arguments in; tool results or denied/error `ToolCall` out. | Central enforcement point for least privilege. |
| `app/tools/calculator.py` | Safe arithmetic and deterministic comparisons. | Numeric expressions, values, options in; calculation, statistics, ranking out. | Used by Analysis; restricted AST rejects code execution nodes. |
| `app/tools/database.py` | Read demo relational records. | Customer/order/complaint IDs and whitelisted table requests in; rows/counts/previews out. | Uses Store query methods and parameterized SQL. |
| `app/tools/documents.py` | Read and ingest local documents. | Relative data/docs path or text in; content, file list, and chunk count out. | Uses VectorMemory; path resolution confines reads. |
| `app/tools/search.py` | Search knowledge, policies, episodes, and optional web. | Query and result limit in; ranked records out. | Uses retrieval helpers; live web is off unless `ALLOW_LIVE_WEB_SEARCH=true`. |
| `app/tools/external_api.py` | Demonstrate side-effect adapters. | Customer/order/action arguments and task context in; action result/reference out. | Writes local action ledger; production replacement point for CRM/email HTTP clients. |

## 3.9 Workers, frontend, and tests

| File | Responsibility | Inputs and outputs | Adjacent interfaces |
|---|---|---|---|
| `app/workers/__init__.py` | Worker package marker. | Python import in; package out. | Supports Celery modules. |
| `app/workers/celery_app.py` | Configure optional Celery application. | Redis URL and settings in; Celery app or `None` out. | Sets JSON serialization, time limits, late acknowledgements, and broker/backend. |
| `app/workers/tasks.py` | Define asynchronous workflow and ingestion jobs. | Task ID/objective/user or text/source/kind in; task result or chunk count out. | Calls runtime and vector memory; exposes direct synchronous stand-ins without Celery. |
| `frontend/__init__.py` | Frontend package marker. | Python import in; package out. | Supports Streamlit module. |
| `frontend/streamlit_app.py` | Human-facing console. | User objectives, approval actions, searches, uploads in; live trace, plans, approvals, memory, stats, and final answer out. | Calls runtime directly and renders state using theme helpers. |
| `frontend/theme.py` | Visual tokens and reusable Streamlit rendering helpers. | State and display values in; styled UI components out. | Used by Streamlit app. |
| `tests/__init__.py` | Test package marker. | Python import in; package out. | Supports pytest discovery. |
| `tests/test_graph.py` | Test graph mechanics. | Graphs, states, interrupts in; assertions for routing, merges, cycles, serialization, and resume. | Verifies engine and state independently of business tools. |
| `tests/test_tools.py` | Test tool safety and correctness. | Tool calls and bad inputs in; assertions for permissions, calculator AST, SQL whitelist, paths, refunds, and replacements. | Protects registry and tool boundaries. |
| `tests/test_workflow.py` | Test end-to-end workflow behavior. | Objectives, seeded data, approvals in; assertions for ordering, escalation, rejection, audit, retrieval, episodic memory, and chunking. | Exercises runtime, nodes, agents, policy, store, and vector memory. |
| `docs/ARCHITECTURE.md` | Existing architecture and viva notes. | Project design in; concise rationale and defense points out. | This guide expands it into an operational manual. |

---

# 4. Setup, Deployment, and Operational Runbook

## 4.1 Prerequisites

### Minimal offline demonstration

- Windows PowerShell, macOS/Linux shell, or equivalent.
- Python 3.10 or newer is recommended because the code uses modern type syntax and dataclasses.
- Internet access only to install Streamlit and pytest; runtime calls do not need an API key or network.
- No Docker, PostgreSQL, ChromaDB, Redis, or Celery is required.

### Full optional profile

- Python 3.10+.
- Docker Desktop or native services for PostgreSQL and Redis.
- A PostgreSQL database and credentials.
- Redis reachable at `REDIS_URL`.
- API key and installed SDK for Anthropic or OpenAI if hosted reasoning is desired.
- A process supervisor/container platform for Uvicorn, Streamlit, and Celery workers.

## 4.2 Environment template

Copy `.env.example` to `.env`. The following table explains every key:

| Key | Meaning | Default / example |
|---|---|---|
| `ANTHROPIC_API_KEY` | Hosted Anthropic credential. | Empty: offline planner. |
| `OPENAI_API_KEY` | Hosted OpenAI credential. | Empty: offline planner. |
| `ANTHROPIC_MODEL` | Anthropic model identifier. | `claude-sonnet-4-5` |
| `OPENAI_MODEL` | OpenAI model identifier. | `gpt-4o-mini` |
| `LLM_PROVIDER` | `auto`, `anthropic`, `openai`, or `simulated`. | `auto` |
| `LLM_TEMPERATURE` | Hosted completion temperature. | `0.2` |
| `LLM_MAX_TOKENS` | Hosted completion output limit. | `1200` |
| `DATABASE_URL` | PostgreSQL connection string. | Empty: SQLite at `data/orbit.db`. |
| `CHROMA_COLLECTION` | Chroma collection name. | `orbit_memory` |
| `EMBEDDING_DIM` | Fallback vector dimension. | `384` |
| `RETRIEVAL_TOP_K` | Default memory result count. | `4` |
| `REDIS_URL` | Redis broker/backend URL. | `redis://localhost:6379/0` |
| `USE_CELERY` | Enable Celery when Redis is reachable. | `false` |
| `APPROVAL_CONFIDENCE_THRESHOLD` | Escalate below this confidence. | `0.65` |
| `MAX_SUPERVISOR_HOPS` | Circuit breaker for supervisor loops. | `12` |
| `TOOL_TIMEOUT_SECONDS` | Tool timeout policy setting. | `20` |
| `ALLOW_LIVE_WEB_SEARCH` | Permit optional live web lookup. | `false` |
| `API_HOST` / `API_PORT` | API bind settings. | `127.0.0.1` / `8000` |
| `ORBIT_DATA_DIR` | Relocate SQLite, vector index, and checkpoints. | `data/` |

Example production-shaped values:

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=replace-me
DATABASE_URL=postgresql://orbit:strong-password@postgres:5432/orbit
REDIS_URL=redis://redis:6379/0
USE_CELERY=true
ORBIT_DATA_DIR=/var/lib/orbit
ALLOW_LIVE_WEB_SEARCH=false
```

Do not commit `.env`. In production, inject secrets through the deployment secret manager.

## 4.3 Local development: offline path

From the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-minimal.txt
python -m app.seed
streamlit run frontend/streamlit_app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-minimal.txt
python -m app.seed
streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501`. Use Mission -> `Support case with an approval gate` -> Run workflow. The first run creates `data/orbit.db`, embedded vector files, and checkpoints as needed.

The convenience launchers perform the seed step automatically:

```powershell
.\run.bat
```

```bash
./run.sh
```

## 4.4 Full Python dependency path

```bash
pip install -r requirements.txt
python -m app.seed
```

Installing the full requirements enables optional components, but capability selection still depends on environment variables and service reachability. For example, ChromaDB is selected when importable; Celery is selected only when `USE_CELERY=true` and Redis responds to `ping()`.

## 4.5 Database initialization

No separate migration command is required for this project. `get_store()` constructs `Store`, and `Store.init_schema()` creates the tables and indexes idempotently.

For PostgreSQL:

1. Create a database and role.
2. Set `DATABASE_URL`.
3. Install `psycopg[binary]`.
4. Run `python -m app.seed` or start any application process; schema creation occurs on first Store use.

The main tables are `tasks`, `agent_runs`, `workflow_states`, `approvals`, `tool_calls`, `action_ledger`, plus demo `customers`, `orders`, and `complaints`. Values use bound parameters. PostgreSQL is the durable relational audit store, while the current graph checkpoint is still file-backed under `ORBIT_DATA_DIR/checkpoints`.

## 4.6 ChromaDB and vector initialization

If `chromadb` is installed, `VectorMemory` selects it and creates/uses `CHROMA_COLLECTION`. Otherwise, it writes the fallback index beneath the configured data directory. Run:

```bash
python -m app.seed
```

This embeds policies, handbook documents, complaints, and other demo knowledge. The API can ingest additional text:

```bash
curl -X POST http://127.0.0.1:8000/memory ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Replacement policy: defective items qualify within 30 days.\",\"source\":\"policy.txt\",\"kind\":\"policy\"}"
```

Use `^` for PowerShell/cmd line continuation only when appropriate; a one-line command is safest. Retrieval returns source-labelled records and uses the configured top-k default.

## 4.7 Redis and Celery

Start Redis using Docker, for example:

```bash
docker run --name orbit-redis -p 6379:6379 -d redis:7-alpine
```

Set:

```dotenv
USE_CELERY=true
REDIS_URL=redis://localhost:6379/0
```

Start a worker from the repository root:

```bash
celery -A app.workers.celery_app.celery_app worker --loglevel=info
```

The runtime will choose Celery only if the package is installed and `settings.has_redis` succeeds. Otherwise it uses the four-thread in-process executor. This fallback is convenient for a lab but is not a substitute for multi-process reliability at production scale.

## 4.8 FastAPI backend

Start the API:

```bash
uvicorn app.api.routes:api --reload --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

Useful endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Version and live/fallback capability report. |
| POST | `/tasks` | Submit a task. |
| GET | `/tasks` | List recent tasks. |
| GET | `/tasks/{task_id}` | Read task, state, agent runs, tool calls, progress. |
| GET | `/approvals` | List pending approvals. |
| POST | `/approvals/{task_id}/approve` | Approve and resume. |
| POST | `/approvals/{task_id}/reject` | Reject and finalize without action. |
| GET | `/tools` | Inspect registered tool permissions and signatures. |
| POST | `/memory` | Ingest text. |
| GET | `/memory/search?q=...` | Search semantic memory. |
| GET | `/stats` | Task and memory counts. |
| GET | `/graph` | Mermaid graph source and backend label. |

## 4.9 Streamlit frontend

Start separately from the API if desired:

```bash
streamlit run frontend/streamlit_app.py
```

The interface provides:

- Mission execution with live node progress.
- Approvals with action summary, arguments, risk, and reason.
- Task detail, traces, plan status, tool calls, and executed actions.
- Memory search and ingestion.
- Architecture view with graph, permissions, and policy rules.

There is no React application in the repository. A React client should use the FastAPI contracts and must add authentication, authorization, polling or WebSocket progress, and a secure reviewer identity flow.

## 4.10 Online versus offline execution

### Cloud-connected mode

Set a provider and install the full requirements. The LLM client sends each role prompt to Anthropic or OpenAI. Optional live web search is enabled only by `ALLOW_LIVE_WEB_SEARCH=true`. Redis/Celery can move long-running tasks off the API process. PostgreSQL and ChromaDB can provide shared persistence and retrieval.

Provider exceptions, including network errors and rate limits, currently fall back to the offline planner. That keeps a demonstration alive, but production policy should decide whether a provider failure is allowed to downgrade silently or must fail closed for regulated actions.

### Air-gapped/offline mode

Leave provider keys empty or set `LLM_PROVIDER=simulated`. ORBIT uses the rule-based offline planner. The fallback vector index is local and needs no service. SQLite, file checkpoints, and the thread pool are local. No live web calls are made by default. A local model server such as Ollama or vLLM is not currently implemented as a provider adapter; adding one requires extending `app/llm/provider.py` and its configuration rather than merely setting an existing variable.

## 4.11 Verification test

Start the API, then run:

PowerShell one-line form:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Expected shape:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "orchestration": "Built-in graph engine",
  "capabilities": {
    "Reasoning": {"mode": "fallback", "detail": "Offline planner (no API key set)"},
    "Orchestration": {"mode": "fallback", "detail": "Built-in graph engine"},
    "Structured store": {"mode": "fallback", "detail": "SQLite (data/orbit.db)"},
    "Semantic memory": {"mode": "fallback", "detail": "Embedded vector index"},
    "Background work": {"mode": "fallback", "detail": "In-process thread pool"}
  }
}
```

The exact capability values change when optional packages, keys, PostgreSQL, ChromaDB, or Redis are available. The invariant health signal is `"status":"ok"`.

Submit and inspect a task:

```powershell
curl.exe -X POST http://127.0.0.1:8000/tasks -H "Content-Type: application/json" -d "{\"objective\":\"Check complaints for CUST-001 and raise a replacement if eligible\",\"background\":false}"
```

Then query the returned ID with `GET /tasks/{task_id}`. A consequential task should reach `awaiting_approval`; approve it through the API and verify an action result and ledger entry.

## 4.12 Automated verification

```bash
python -m pytest tests/ -q
```

The test suite covers graph mechanics, interrupts and resume, permission enforcement, calculator safety, SQL/table rules, path confinement, action integrity, plan ordering, approval behavior, audit traces, retrieval, episodic memory, and chunking.

---

# 5. Technical Defense and Viva Voce Manual

## 5.1 Architectural concepts

**Q: What is an AI agent?**  
A: In ORBIT, an agent is a role-specific program wrapper around model reasoning, state, and tools. It can inspect assigned context, call permitted tools, return structured output, and continue as part of a workflow. It is not an autonomous person and cannot bypass the registry or policy.

**Q: Why use multiple agents instead of one large model?**  
A: Multiple roles reduce context mixing, make failures easier to locate, and allow least-privilege tools. Research can read data without being able to issue refunds. The tradeoff is orchestration overhead and extra model calls, so the design is worthwhile when the task has distinct stages or real side effects.

**Q: What is orchestration?**  
A: Orchestration is the control logic that connects state, agents, tools, routing, checkpoints, and stopping conditions. It turns separate calls into one auditable case lifecycle.

**Q: What is the difference between a chain and a state graph?**  
A: A chain usually moves forward through a fixed sequence. A state graph stores a shared case, can choose different branches, can loop back to a supervisor, and can pause for an external decision. ORBIT needs the latter because action approval and recovery are control-flow events.

**Q: Is ORBIT acyclic or cyclic?**  
A: The topology contains intentional cycles: Research and Analysis return to Supervisor, and Execute returns to Supervisor. It is not an uncontrolled infinite loop because `MAX_SUPERVISOR_HOPS` and `CompiledGraph.invoke(max_steps=80)` provide circuit breakers.

**Q: Why does the Supervisor have no tools?**  
A: Coordination and execution are separated. The Supervisor decides who should work and writes the final answer; specialists own capabilities. This reduces the blast radius of a routing mistake.

**Q: Why have both a built-in engine and LangGraph?**  
A: The built-in engine makes node contracts, merging, interrupt behavior, and file checkpoints inspectable with minimal dependencies. The functions use a portable node/state/conditional-edge shape, so a future LangGraph adapter can reuse them. The current code does not claim that LangGraph is executing the graph.

**Q: What is the shared state?**  
A: It is a typed case file: objective, plan, cursor, hops, evidence, analysis, validation, pending action, approval, trace, errors, and answer. A node returns only changed fields, which makes updates and checkpoints easy to inspect.

## 5.2 Memory and retrieval

**Q: Why use SQL state and vector memory together?**  
A: SQL is for exact, transactional questions: task status, approval status, audit records, joins, and action ledger. Vectors are for meaning-based recall: finding a relevant policy paragraph even when the query uses different words. They solve different lookup problems.

**Q: What is RAG in ORBIT?**  
A: Retrieval-Augmented Generation means retrieving relevant policy, document, complaint, or episode records, labelling them with their source, and placing them in the agent's prompt before it reasons. Retrieval supplies evidence; it does not replace validation.

**Q: How does fallback retrieval work?**  
A: Documents are split by paragraphs and size. The fallback index creates normalized hashed-token vectors, adds lexical overlap and IDF weighting, and ranks matches. ChromaDB can replace this small-corpus implementation when installed.

**Q: What is vector drift?**  
A: Vector drift occurs when the embedding model or corpus changes so old vectors no longer have comparable meaning. A production system should version embeddings, reindex on model changes, monitor retrieval quality, and preserve source/version metadata. The current fallback index should be rebuilt when its embedding scheme changes.

**Q: How should chunking be improved at scale?**  
A: Keep chunks small enough to focus retrieval but large enough to preserve a policy rule's conditions and exceptions. Split on headings and paragraphs, retain document and section metadata, use overlap where needed, and evaluate recall with representative questions. The current implementation is deliberately simple for a small demonstration corpus.

**Q: Is a completed task memory or RAG?**  
A: RAG is the retrieval technique. Memory is the broader ability to retain information. ORBIT uses vector retrieval both for static knowledge and for completed tasks stored as `episode` records.

## 5.3 Security and safety

**Q: How is least privilege enforced?**  
A: Each tool declares `allowed_agents`, required parameters, and a risk level. `ToolRegistry.execute()` checks the calling agent before function dispatch. Prompts are not the security boundary.

**Q: How is SQL injection reduced?**  
A: Values are bound parameters rather than interpolated strings, and dynamic table access is restricted to a whitelist. Production code should retain this pattern and add database roles with read-only permissions for research tools.

**Q: How is the calculator prevented from becoming code execution?**  
A: It parses an AST and accepts only arithmetic structures. Imports, calls, attributes, and other executable nodes are rejected.

**Q: How are file reads confined?**  
A: Document paths are resolved and checked against approved `data/` and `docs/` locations. The tool does not treat arbitrary user-provided paths as trusted.

**Q: Can a model directly issue a refund?**  
A: Not through the intended path. The model proposes a tool and arguments; the registry checks the agent, policy assesses risk, validation runs, and the human gate pauses high-risk actions. Only the approved proposal reaches execution.

**Q: What prevents infinite recursion?**  
A: The supervisor hop limit and graph max-step limit. When the hop limit is reached, pending plan steps are skipped and validation/finalization closes the run.

**Q: Is the reviewer authenticated?**  
A: No. `decided_by` is recorded but not verified. Authentication, authorization, reviewer identity, CSRF protection, rate limiting, and audit-log tamper resistance are production requirements.

## 5.4 Error handling and recovery

**Q: What happens on an LLM rate limit or network failure?**  
A: `LLMClient.complete()` catches provider exceptions and uses the offline planner, labelling the result `simulated` with a fallback marker. For production, decide whether a risky task should fail closed rather than downgrade.

**Q: What happens when model JSON is malformed?**  
A: `complete_json()` strips code fences and searches for a JSON object or array. If parsing still fails, a typed fallback supplied by the caller is returned; otherwise an `LLMError` is raised.

**Q: What happens when a tool is denied or fails?**  
A: The call is recorded with status/error, the agent reports it in its output, and Validation sees unresolved issues. Confidence can fall and the policy can require review.

**Q: What happens if the process restarts during approval?**  
A: The file checkpoint contains the serialized state and next node. `get_workflow().get_state(task_id)` reloads it, and `resume()` re-enters the gate with the reviewer's decision.

**Q: What happens if the database connection is lost?**  
A: Store operations can raise and the runtime marks an unhandled workflow failure. Graph checkpoints are separate, so a production operator can recover state, but this repository does not implement a full database outage queue or transactionally unified checkpoint.

**Q: Does Celery retry?**  
A: The Celery workflow task is configured with up to two retries and a ten-second countdown. The direct/thread fallback does not provide distributed retry semantics.

## 5.5 Business, scalability, and impact

**Q: How does decomposition reduce cost?**  
A: It can reduce the context sent to each call and lets deterministic tools handle arithmetic and lookup. It can also increase cost because several specialist calls are made. The correct metric is end-to-end quality and token cost per completed business outcome, not call count alone.

**Q: How does ORBIT handle latency?**  
A: The API can return a task ID immediately. Celery and Redis support distributed workers; without them, a thread pool provides a local fallback. UI streaming reports progress. A production deployment should add queue metrics, timeouts, cancellation, and client backoff.

**Q: How would it scale?**  
A: Move task execution to multiple Celery workers, use PostgreSQL as shared relational storage, use a shared durable checkpoint implementation, use a managed vector service or shared Chroma deployment, and put the API behind an authenticated gateway. Add idempotency keys before allowing real external effects.

**Q: What is the most important scalability limitation today?**  
A: Graph checkpoints are local JSON files, and the fallback vector index is local. Multiple application replicas cannot safely coordinate from those files without shared storage or a database-backed checkpointer.

**Q: What is ORBIT's SDG alignment?**  
A: SDG 9 is primary because the project demonstrates intelligent digital infrastructure for reliable workflow automation. SDG 8 is secondary through productivity and structured work assistance. SDG 16 is secondary through human oversight, traceability, and accountable decisions. These are design-alignment claims, not impact measurements.

**Q: What is the project's innovation?**  
A: The defensible contribution is the combination of role decomposition, scoped permissions, evidence retrieval, durable state, policy-coded approval, auditable action handling, and graceful offline operation in one inspectable workflow. It is not merely the use of multiple models.

---

# 6. Design Limits and Production Hardening Checklist

## 6.1 Current limitations to state openly

- LangGraph is optional and not the current executor.
- PostgreSQL is optional; SQLite is the default. The graph checkpointer is file-backed even on PostgreSQL.
- ChromaDB is optional; the fallback uses hashed-token embeddings and is intended for a small corpus.
- Redis/Celery are optional; the fallback is an in-process thread pool.
- External API functions write a local action ledger; they do not call a live CRM, email service, or payment processor.
- There is no authentication or verified reviewer identity.
- There is no React frontend; Streamlit is the implemented UI.
- Ollama/vLLM is not an existing provider adapter.
- Provider failures fall back to simulated reasoning, which may be unacceptable for regulated actions.
- File checkpoints and local vector files require shared storage or replacement for multi-replica production deployment.

## 6.2 Recommended production hardening

1. Add authentication and role-based authorization to API and approval routes.
2. Replace local action adapters with allow-listed, authenticated clients using idempotency keys and outbound timeouts.
3. Fail closed for high-risk actions when hosted-model, validation, policy, or persistence guarantees are unavailable.
4. Implement a PostgreSQL-backed checkpointer or a shared object-store checkpoint with locking and retention.
5. Add schema migrations, connection pooling, backup/restore tests, and encryption at rest.
6. Add distributed locks so two reviewers cannot resume the same task concurrently.
7. Add structured logging, metrics, traces, queue depth alerts, and data retention rules.
8. Version embedding models and reindex on changes; evaluate retrieval with a test set.
9. Add prompt-injection tests for documents, web results, and tool arguments.
10. Add cancellation, timeout propagation, retry classification, dead-letter handling, and idempotent task submission.
11. Use a secrets manager and rotate provider/database credentials.
12. Add redaction for personal data in logs and define who may view approval arguments.

---

# 7. Appendix: API and Data Contracts

## 7.1 Health response

```json
{
  "status": "ok",
  "version": "1.0.0",
  "orchestration": "Built-in graph engine",
  "capabilities": {
    "Reasoning": {"mode": "fallback", "detail": "Offline planner (no API key set)"},
    "Orchestration": {"mode": "fallback", "detail": "Built-in graph engine"},
    "Structured store": {"mode": "fallback", "detail": "SQLite (data/orbit.db)"},
    "Semantic memory": {"mode": "fallback", "detail": "Embedded vector index"},
    "Background work": {"mode": "fallback", "detail": "In-process thread pool"}
  }
}
```

## 7.2 Task creation contract

Request fields:

```json
{
  "objective": "string, 3-4000 characters",
  "user_id": "string, default api-user",
  "background": "boolean, default true"
}
```

Response:

```json
{"task_id":"task-...","status":"accepted"}
```

## 7.3 Approval decision contract

Request:

```json
{"decided_by":"reviewer","note":"Evidence checked; approve."}
```

Response:

```json
{"task_id":"task-...","decision":"approved"}
```

The decision value is `approved` or `rejected`. A conflict or missing checkpoint produces an HTTP 409 from the API.

## 7.4 Conceptual state contract

```text
task_id, objective, user_id, status
plan[], cursor, hops
research{}, analysis{}, validation{}, action_result{}
retrieved[]
risk, confidence, approval, pending_action
 events[], tool_calls[], errors[]
final_answer, created_at, updated_at
```

This state is the central explanation for ORBIT's behavior. Every visible answer should be traceable to a state transition, an agent result, a tool observation, a policy decision, or a human decision.

---

## Closing statement

ORBIT is best understood as a controlled case-management workflow with AI specialists, not as a chatbot with a longer prompt. Its central engineering idea is simple: make reasoning decomposable, make permissions explicit, make side effects interruptible, and make the whole journey inspectable. The repository already demonstrates those ideas locally; production deployment must add the identity, shared durability, external integrations, and operational controls listed above.
