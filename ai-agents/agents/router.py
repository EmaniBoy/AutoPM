from __future__ import annotations
import json
from typing import Dict, Any
from agents.shared import nv_chat

ROUTER_SYSTEM_PROMPT = """You are the Orchestrator for a Product Manager’s AI assistant.

Available agents:
- RESEARCH: ingest & summarize docs, extract issues/opportunities, propose product ideas.
- JIRA: turn problems/opportunities into epics/stories with INVEST acceptance criteria; link evidence.
- DEVOPS: generate/upgrade a GitHub Actions workflow and a 3-week sprint plan (capacity, dependencies, risks).

Output ONLY valid JSON with:
{
  "intent": "<one of: research | jira | devops | research+jira | jira+devops | research+jira+devops | status | other>",
  "inputs": { ... },
  "apply_changes": true|false
}

Rules:
- If the user mentions files/docs/PDFs/slides or “summarize / analyze / insights / ideas”, start with "research".
- Stories/epics/INVEST/Jira/backlog -> "jira".
- CI/CD/workflow/sprint plan/GitHub Actions/release -> "devops".
- Chain in order for end-to-end (research -> jira -> devops).
- Set apply_changes=true only if user clearly asks to open PRs / create Jira issues / “go ahead and apply”.
- Keep inputs minimal but useful (repo, project_key, capacity, sources, time window, etc.).
"""

def plan_from_text(user_text: str) -> Dict[str, Any]:
    user = f"User request:\n{user_text}\n\nReturn ONLY the JSON object."
    out = nv_chat(
        [{"role":"system","content":ROUTER_SYSTEM_PROMPT},
         {"role":"user","content":user}],
        temperature=0.1,
        max_tokens=400
    )
    # be defensive: strip code fences if the model adds them
    out = out.strip()
    if out.startswith("```"):
        out = out.strip("`").split("\n",1)[-1]
    plan = json.loads(out)
    # tiny sanitize
    intent = plan.get("intent","other").lower()
    allowed = {"research","jira","devops","research+jira","jira+devops","research+jira+devops","status","other"}
    if intent not in allowed:
        plan["intent"] = "other"
    if "apply_changes" not in plan:
        plan["apply_changes"] = False
    if "inputs" not in plan:
        plan["inputs"] = {}
    return plan
