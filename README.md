# ORBIT

**Orchestrated Reasoning & Behavioral Intelligence Technology**

A multi-agent AI platform with tool use, persistent memory, asynchronous
execution and human-in-the-loop escalation.

Minor project · School of Engineering & Technology, Vivekananda Institute of
Professional Studies – Technical Campus
Supervisor: Dr. Vikas Badgujar
Gurvesh Aryan (05317702723) · Ankit Chauhan (06317702724) · Tanjal Kumar (06717702724)

---

## What it does

A single LLM answering a question is one thing. A complex objective —
*"check this customer's complaint history, work out whether they qualify for a
replacement under our policy, and raise the request if they do"* — is another.
It needs retrieval, reasoning, a policy check, and an action with real
consequences.

ORBIT handles that by splitting the work across specialist agents under a
Supervisor, giving each only the tools its job requires, and **stopping for a
human before anything consequential happens.**

```
 objective
     │
     ▼
 Supervisor ──┬── Research  ── memory, documents, database
              ├── Analysis  ── calculation, comparison, judgement
              └── Action    ── external APIs
                    │
                    ▼
                Validation ── audits grounding and consistency
                    │
          action staged? ──── no ──▶ final response
                    │ yes
                    ▼
          ⏸  HUMAN APPROVAL  ⏸
              │           │
          approve      reject
              │           │
           execute    stop, explain
```

The pause is real. The workflow is checkpointed to disk mid-execution, so a
paused task survives restarting the application.

---

## Quick start

```bash
cd orbit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-minimal.txt
python -m app.seed
streamlit run frontend/streamlit_app.py
```

Then open <http://localhost:8501>, pick **"Support case with an approval
gate"** on the Mission tab, and press **Run workflow**.

No API key, database or message broker is needed to run any of this. See
*Graceful degradation* below.

### Optional: live model reasoning

```bash
cp .env.example .env
# add ANTHROPIC_API_KEY=... or OPENAI_API_KEY=...
pip install -r requirements.txt
```

---

## Graceful degradation

Every external dependency named in the synopsis is optional at runtime. ORBIT
detects what is present, uses it, and substitutes a built-in equivalent
otherwise. The sidebar reports which of each is in play, so you always know
whether a result came from a real model or the offline planner.

| Component | Used when available | Fallback |
|---|---|---|
| Reasoning | Anthropic / OpenAI | Rule-based offline planner |
| Orchestration | LangGraph | Built-in graph engine (`app/graph/engine.py`) |
| Structured store | PostgreSQL | SQLite at `data/orbit.db` |
| Semantic memory | ChromaDB | Embedded vector index |
| Background work | Celery + Redis | In-process thread pool |

This is deliberate. A demonstration that dies because Redis is not running
teaches nobody anything, and an examiner can unplug any single component and
watch the platform degrade rather than fail.

Anything produced by the offline planner is labelled *simulated* in the UI.
It is never presented as model reasoning.

---

## The five agents

| Agent | Responsibility | Tools |
|---|---|---|
| **Supervisor** | Decomposes the objective, routes each step, writes the final answer | none — it coordinates, it does not work |
| **Research** | Gathers evidence from memory, documents and the database | `knowledge_search`, `policy_lookup`, `lookup_customer`, `customer_complaints`, `customer_orders`, `read_document`, … |
| **Analysis** | Compares, calculates, concludes, reports calibrated confidence | `calculate`, `summarise_numbers`, `compare_options` |
| **Action** | Executes one approved external operation | `create_replacement_request`, `issue_refund`, `send_email`, `create_ticket`, `update_record` |
| **Validation** | Audits grounding and consistency before a human is asked | none — an auditor that can fetch evidence fixes problems instead of reporting them |

Permissions are enforced in Python at call time, not in a prompt. If the
Research Agent asks for `issue_refund`, the registry returns a *denied* call
before the function is ever looked up.

---

## When a human is asked

The escalation policy lives in `app/policy.py` as ordinary code. A policy
written as an instruction to a model can be argued out of; a policy written as
a function cannot. A workflow pauses when any of these hold:

- the proposed tool is classified **high** risk;
- confidence falls below the threshold (default 65%);
- the Validation Agent left unresolved issues;
- a monetary amount exceeds the ceiling;
- the action sends something to an external recipient;
- required arguments for the call are missing.

Every approval records what was proposed, why you were asked, who decided, and
what note they left.

---

## Running the other interfaces

```bash
# Terminal — a fallback if the projector or browser misbehaves
python -m app.cli --status
python -m app.cli "Check complaints for CUST-001 and raise a replacement if eligible"
python -m app.cli --pending
python -m app.cli --approve task-xxxxxxxx --by "Dr. Badgujar" --note "defect confirmed"

# REST API
uvicorn app.api.routes:api --reload --port 8000    # docs at /docs

# Celery worker (needs Redis and USE_CELERY=true)
celery -A app.workers.celery_app.celery_app worker --loglevel=info

# Tests
python -m pytest tests/ -q
```

---

## Project layout

```
orbit/
├── app/
│   ├── config.py              settings and capability detection
│   ├── policy.py              the escalation rules
│   ├── runtime.py             service layer used by every front end
│   ├── seed.py                demo dataset and knowledge base
│   ├── cli.py                 terminal interface
│   ├── agents/                supervisor, research, analysis, action, validation
│   ├── graph/
│   │   ├── state.py           the shared workflow state
│   │   ├── engine.py          graph execution, checkpoints, interrupts
│   │   ├── nodes.py           node functions and routers
│   │   └── workflow.py        graph assembly
│   ├── tools/                 registry, permissions, and the tools
│   ├── memory/                structured store, vector store, retrieval
│   ├── workers/               Celery app and background tasks
│   └── api/routes.py          FastAPI backend
├── frontend/
│   ├── streamlit_app.py       the console
│   └── theme.py               visual language and components
├── tests/                     51 tests
├── docs/                      architecture notes and viva preparation
└── data/                      databases, vector index, checkpoints (gitignored)
```

---

## Demonstration script

Five minutes, in this order:

1. **Mission → "Support case with an approval gate" → Run workflow.**
   Watch the plan appear and the trace fill in: supervisor routes to research,
   research calls four tools, analysis concludes, validation audits, and the
   workflow stops.
2. **Read the amber panel.** It states the action in plain language, why you
   are being asked, and the exact arguments that will be used.
3. **Approve.** The workflow resumes from the checkpoint and executes. Check
   the reference number in Tasks → *Executed actions*.
4. **Run it again and reject.** The ledger stays unchanged; the user still
   gets an explanation.
5. **Architecture tab.** Show the graph, the per-agent tool permissions, and
   the escalation policy.

For the durability claim: pause a task, stop Streamlit entirely, start it
again, and find the task still waiting on the Approvals tab.

---

## Known limitations

- Without an API key, reasoning is heuristic rather than a language model.
  Every other subsystem is real.
- External actions are recorded in a local audit ledger rather than sent to a
  live CRM. Swapping `_dispatch` in `app/tools/external_api.py` for an HTTP
  client is the only change needed; the approval gate and permission model are
  already genuine.
- Retrieval uses hashed-token embeddings in the fallback path. Quality is good
  over a small corpus and would benefit from a sentence-transformer at scale —
  replace `embed()` in `app/memory/vector.py`.
- There is no authentication. The reviewer name in the sidebar is recorded,
  not verified.

---

## SDG alignment

Primarily **SDG 9 — Industry, Innovation and Infrastructure**: intelligent
digital infrastructure for complex workflow automation. Secondarily **SDG 8**
through productivity gains, and **SDG 16** through the accountability that
enforced human oversight and a complete audit trail provide.
