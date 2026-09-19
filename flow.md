1. **Data is already there**
   - The project starts with a small demo dataset in memory and in the SQLite database. It has customers, orders, and complaints. The data is created by `app.seed`.
   - `CUSTOMERS` contains 5 customer records. `ORDERS` contains 7 orders. `COMPLAINTS` contains 5 complaint records. These rows are inserted into the `customers`, `orders`, and `complaints` tables in the store.
   - It also loads policy and support documents into semantic memory. These are not normal database rows; they are text chunks stored in the vector memory so the system can search for similar text later.
   - This means the system already has examples to work with before a user even types an objective.

2. **Program starts**
   - The app is started from the terminal with `python -m app.cli` or from Streamlit with `streamlit run frontend/streamlit_app.py`.
   - On startup, the UI checks whether the database is empty and, if it is, it runs `seed_all()`. The script fills in the demo data if needed.
   - If a user gives an objective, the code creates an `OrbitState` object. This object holds the task id, the user objective, a plan, event log, research results, analysis result, approval, and final answer.
   - The task is saved in the `tasks` table right away.

3. **Data is read**
   - The real work begins in `app.runtime._run_workflow()`. It gets the workflow graph and calls it with the current `OrbitState`.
   - The first node is `plan_node`. This node calls `SupervisorAgent.plan()`. The supervisor does not read the raw database directly first. It asks for memory context using `retrieve_context()` and `recall_similar_tasks()`.
   - The supervisor builds a prompt containing the user objective and any retrieved memory. Then it asks the model for a list of steps. If the model fails, it falls back to a simple plan with research and analysis.
   - The plan is turned into `PlanStep` objects: each step has an agent name like `research`, `analysis`, or `action`, an instruction, and a status.

4. **Data is changed and processed**
   - The workflow then follows the plan. `research_node` calls `ResearchAgent.run()`. This agent looks at the objective and picks tools. It uses keyword rules, such as customer ids, order ids, policy words, complaint history words, and email patterns.
   - The research agent calls tools like `lookup_customer`, `customer_orders`, `customer_complaints`, `policy_lookup`, `knowledge_search`, and `web_search`. These tools read from the already-filled database or memory and return rows or text chunks.
   - The tool results are combined into one prompt for the model. The model returns a summary, findings, sources, and coverage. That result is saved into `state.research`.
   - Next comes `analysis_node`, which reads the findings and runs a numeric check on any numbers found. It calls `summarise_numbers` from the calculator tool if numbers are present, then asks the model for a conclusion and a confidence value.
   - The conclusion and confidence are saved into `state.analysis`.

5. **Model/function runs**
   - After research and analysis, the workflow reaches `validation_node`. This node runs the `ValidationAgent`. It checks basic rules: did research find anything, did it include sources, is there a conclusion, did any tool fail, and did the objective terms appear in the produced text?
   - The validator builds a list of issues and reduces the confidence if the output looks weak. It stores the final validation result in `state.validation`.
   - The next real decision point is `action_propose_node`, which creates a proposed action. It picks one tool such as `create_replacement_request`, `issue_refund`, `send_email`, `create_ticket`, or `update_record`.
   - Immediately after that, `policy.assess_action()` checks risk. It marks the action as needing approval if the tool is high risk, confidence is too low, money is above a ceiling, or an external message is involved.
   - If approval is required, the code creates an `ApprovalRequest` and the workflow pauses at `human_gate_node` with `state.status = "awaiting_approval"`.

6. **Result is produced**
   - The approval gate is the real stop point. The user can approve or reject the action. If they approve, the graph resumes and calls `action_execute_node`. This executes the exact tool that was proposed, such as writing a replacement request or issuing a refund.
   - The external tools do not hit a real external system. They write a local record into `action_ledger` and return a result. The code in `app.tools.external_api.py` records what happened and stores it with the task id.
   - After execution, `supervisor.synthesise()` builds a final prompt using research, analysis, validation, action result, and approval info. The model writes the final answer. `finalise_node` stores it in `state.final_answer`, saves the final state to the database, and writes the task status to `completed` or `rejected`.
   - It also stores a short summary in the vector memory as an episode, so future tasks may recall similar earlier work.

7. **Result is shown or used**
   - The final answer is shown in the Streamlit UI in the “Mission” tab and also printed in the terminal CLI. The UI reads the task state and displays the plan, trace, research findings, analysis, validation, and tool-call details.
   - The database stores the task, the agent runs, the tool calls, the approval, and the workflow checkpoints. The memory store also keeps the text knowledge and the short episodes.
   - So the project is not one big model run. It is a small loop: read demo data, plan, gather evidence, reason, validate, ask for approval if the action is risky, execute the safe action, then write the final answer and save records.
   - In plain words: the code already has support-case data, moves it through several known steps, changes it into findings and decisions, sometimes pauses for a person, and finally produces a final written answer plus saved records.
