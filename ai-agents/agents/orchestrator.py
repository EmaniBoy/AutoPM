# ai-agents/agents/orchestrator.py
from __future__ import annotations

import os, json, requests
from typing import TypedDict, Dict, Any, List, Optional

from dotenv import load_dotenv, find_dotenv

# Load .env deterministically (ai-agents/.env + fallback search)
_THIS_DIR = os.path.dirname(__file__)
_ENV_PATH = os.path.abspath(os.path.join(_THIS_DIR, "..", ".env"))
if os.path.isfile(_ENV_PATH):
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
auto_env = find_dotenv(usecwd=True)
if auto_env:
    load_dotenv(dotenv_path=auto_env, override=False)

# ---------- LangGraph ----------
from langgraph.graph import StateGraph, END

# ---------- Router (LLM that makes a plan) ----------
from agents.router import plan_from_text

# ---------- Shared + DevOps Planner ----------
from agents.shared import (
    GITHUB_REPO,
    GITHUB_DEFAULT_BRANCH,
    gh_comment_issue,
    nv_chat,
)
from agents.devops_planner import (
    detect_repo_state,
    propose_plan,
    format_sprint_markdown,
    upsert_workflow,  # expects (filename, yaml_text, branch, sprint_md)
)


# ===================== State =====================
class OrchestratorState(TypedDict, total=False):
    # Inputs / context
    repo: str
    project_key: str
    team_context: Any
    raw_input: str  # freeform PM text; also passed into Jira agent
    sources: List[str]
    capacity_default: int

    # Router
    user_text: str
    router_plan: Dict[str, Any]
    auto_apply: bool

    # Research output
    backlog: List[Dict[str, Any]]  # compact issues/problems for Jira/DevOps

    # DevOps output
    repo_summary: Dict[str, Any]
    plan: Dict[str, Any]           # YAML + sprint object (from planner)
    sprint_md: str
    pr_url: Optional[str]

    # UX / logging
    summary_md: str
    errors: List[str]


# ===================== Nodes =====================

def node_router(state: OrchestratorState) -> OrchestratorState:
    """
    Read PM natural language (user_text/raw_input) -> JSON plan:
      intent ∈ {research | jira | devops | research+jira | jira+devops | research+jira+devops | status | other}
      inputs {project_key, repo, capacity_default, sources, ...}
      apply_changes: bool
    """
    msg = state.get("user_text") or state.get("raw_input") or "Plan next sprint."
    try:
        plan = plan_from_text(msg)
        state["router_plan"] = plan

        # Let router set apply flag; CLI --apply can still force it later
        state["auto_apply"] = bool(plan.get("apply_changes", state.get("auto_apply", False)))

        # Fill defaults from router inputs, if present
        inputs = plan.get("inputs", {})
        state["project_key"] = inputs.get("project_key", state.get("project_key", "PB"))
        state["team_context"] = inputs.get("team_context", state.get("team_context", "AutoPM"))
        state["raw_input"]    = inputs.get("raw_input", state.get("raw_input", msg))
        state["repo"]         = inputs.get("repo", state.get("repo", GITHUB_REPO))
        state["capacity_default"] = int(inputs.get("capacity_default", state.get("capacity_default", 12)))
        state["sources"] = inputs.get("sources", state.get("sources", []))
    except Exception as e:
        state.setdefault("errors", []).append(f"router: {e}")
        state["router_plan"] = {"intent": "other", "inputs": {}, "apply_changes": False}
    return state


def _run_research(state: OrchestratorState) -> None:
    """
    RESEARCH agent placeholder.
    Replace this with your Nemotron Parse API + RAG pipeline.
    For now: extract issues/ideas as concise bullets so Jira has something.
    """
    try:
        prompt = (
            "Extract 5 problems/opportunities and 3 product ideas from the context. "
            "Return concise dash bullets."
        )
        summary = nv_chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Context: {state.get('raw_input')}\nSources: {state.get('sources')}"}
            ],
            max_tokens=500,
        )
        # Seed a small backlog from bullets
        items = [
            {"summary": line.strip("- ").strip()}
            for line in summary.splitlines()
            if line.strip().startswith("-")
        ][:8]
        state["backlog"] = items or state.get("backlog", [])
        state.setdefault("summary_md", "")
        state["summary_md"] += "### Research Findings\n" + summary + "\n\n"
    except Exception as e:
        state.setdefault("errors", []).append(f"research: {e}")


def _run_jira(state: OrchestratorState) -> None:
    """
    Jira Storywriter Agent:
      - Try POST {JIRA_AI_URL}/jiraai/full-run (E2E create)
      - Fallback POST {JIRA_AI_URL}/jiraai/fullrun (alt route) and /jiraai/generate
      - Normalize to compact items and render clickable links using JIRA_BASE_URL
      - Hard-code projectKey='PB' for reliable local demo
    """
    base = os.getenv("JIRA_AI_URL", "http://localhost:4000").rstrip("/")
    jira_base = os.getenv("JIRA_BASE_URL", "").rstrip("/")

    # Hard-code PB so orchestrator always sends a valid key (like your working curl)
    project_key = "PB"

    payload = {
        "projectKey": project_key,
        "teamContext": state.get("team_context", "AutoPM team"),
        "rawInput": state.get("raw_input", "Create INVEST stories."),
    }

    # helpful debug
    print("[JIRA DEBUG] POST", f"{base}/jiraai/full-run", "payload=", json.dumps(payload))

    def _post(path: str):
        return requests.post(f"{base}{path}", json=payload, timeout=60)

    try:
        # Try hyphen route, then no-hyphen, then generate
        data = None
        errors = []

        r = _post("/jiraai/full-run")
        if r.status_code < 400:
            data = r.json()
        else:
            errors.append(f"/full-run -> {r.status_code}: {(r.text or '')[:1000]}")
            r2 = _post("/jiraai/fullrun")
            if r2.status_code < 400:
                data = r2.json()
            else:
                errors.append(f"/fullrun  -> {r2.status_code}: {(r2.text or '')[:1000]}")
                r3 = _post("/jiraai/generate")
                if r3.status_code < 400:
                    data = r3.json()
                else:
                    errors.append(f"/generate -> {r3.status_code}: {(r3.text or '')[:1000]}")

        if data is None:
            raise RuntimeError("Jira service failed.\n" + "\n".join(errors) + f"\nPayload: {json.dumps(payload)}")

        # Collect items
        items: List[Dict[str, Any]] = []
        if isinstance(data.get("stories"), list):
            items.extend(data["stories"])

        # Epics as dict {name: KEY}
        if isinstance(data.get("epics"), dict):
            for name, key in data["epics"].items():
                items.append({"key": key, "summary": name, "type": "Epic"})

        # Generic buckets
        for k in ("backlog", "issues", "items"):
            v = data.get(k)
            if isinstance(v, list):
                items.extend(v)

        # Normalize + linkify
        normalized: List[Dict[str, Any]] = []
        for it in items:
            key = it.get("key")
            summary = it.get("summary") or it.get("title") or ""
            issue_type = it.get("type") or it.get("issueType")
            url = f"{jira_base}/browse/{key}" if (jira_base and key) else None
            normalized.append({"key": key, "summary": summary, "type": issue_type, "url": url})

        # Keep top N
        top_n = int(os.getenv("JIRA_TOP_LIMIT", "10"))
        state["backlog"] = normalized[:top_n] or state.get("backlog", [])

        # Human-readable summary
        state.setdefault("summary_md", "")
        if state["backlog"]:
            lines = []
            for i in state["backlog"][:8]:
                if i.get("url") and i.get("key"):
                    lines.append(f"- [{i['key']}]({i['url']}) — {i.get('summary','')}")
                elif i.get("key"):
                    lines.append(f"- {i['key']} — {i.get('summary','')}")
                else:
                    lines.append(f"- {i.get('summary','')}")
            state["summary_md"] += "### Jira: generated backlog\n" + "\n".join(lines) + "\n\n"
        else:
            state["summary_md"] += "### Jira: generated backlog\n_no items returned_\n\n"

    except Exception as e:
        # Surface error but keep progress so DevOps can still run
        state.setdefault("errors", []).append(f"jira: {e}")
        state.setdefault("summary_md", "")
        state["summary_md"] += "### Jira: failed (see errors)\n\n"
        state["backlog"] = state.get("backlog", [])

def _run_devops(state: OrchestratorState) -> None:
    """
    DEVOPS planner: detect repo → ask LLM for CI YAML & 3-week sprint → optionally apply (PR).
    """
    try:
        summary = detect_repo_state()
        # pass compact Jira context if available
        if state.get("backlog"):
            compact = []
            for it in state["backlog"]:
                compact.append({
                    "key": it.get("key") or it.get("id"),
                    "type": it.get("type") or it.get("issueType"),
                    "summary": (it.get("summary") or it.get("title") or "")[:200],
                    "estimate": it.get("estimate") or it.get("storyPoints"),
                    "dependsOn": it.get("dependsOn") or it.get("dependencies"),
                })
            summary["jira_top"] = compact

        plan = propose_plan(summary)
        state["plan"] = plan
        state["sprint_md"] = format_sprint_markdown(plan.get("sprint", {}))

        # Maybe apply (create/patch workflow + open PR)
        if state.get("auto_apply"):
            fname = (plan.get("filename") or "ci.yml").strip()
            yaml_text = plan.get("yaml") or ""
            res = upsert_workflow(fname, yaml_text, GITHUB_DEFAULT_BRANCH, state.get("sprint_md", ""))
            pr = res.get("pr") or {}
            state["pr_url"] = pr.get("html_url")
            # Optional: comment sprint plan on the PR
            if pr.get("number"):
                gh_comment_issue(GITHUB_REPO, pr["number"], f"## Sprint Plan\n\n{state['sprint_md']}")

        state.setdefault("summary_md", "")
        pr_line = f"PR: {state.get('pr_url')}" if state.get("pr_url") else "PR: (preview only)"
        state["summary_md"] += f"### DevOps: CI + Sprint\n{pr_line}\n\n"
    except Exception as e:
        state.setdefault("errors", []).append(f"devops: {e}")


def node_dispatch(state: OrchestratorState) -> OrchestratorState:
    """
    Execute agents according to the router's plan.
    """
    intent = (state.get("router_plan", {}).get("intent") or "").lower()
    if intent in ("", "other", "status"):
        state.setdefault("summary_md", "")
        state["summary_md"] += "No actionable intent detected.\n"
        return state

    if intent.startswith("research"):
        _run_research(state)
    if "jira" in intent:
        _run_jira(state)
    if "devops" in intent:
        _run_devops(state)
    return state

def node_jira_storywriter(state: OrchestratorState) -> OrchestratorState:
    base = os.getenv("JIRA_AI_URL", "http://localhost:4000").rstrip("/")
    jira_base = os.getenv("JIRA_BASE_URL", "").rstrip("/")
    payload = {
        "projectKey": state.get("project_key", "PB"),
        "teamContext": state.get("team_context", "AutoPM context"),
        "rawInput": state.get("raw_input", "Plan the next sprint."),
    }

    def _call(path: str):
        url = f"{base}{path}"
        r = requests.post(url, json=payload, timeout=60)
        return r

    try:
        r = _call("/jiraai/full-run")
        if r.status_code >= 400:
            # fall back to /generate with explicit error context
            body = r.text
            r2 = _call("/jiraai/generate")
            if r2.status_code >= 400:
                raise RuntimeError(f"/full-run -> {r.status_code} {body[:1000]}; /generate -> {r2.status_code} {r2.text[:1000]}")
            data = r2.json()
        else:
            data = r.json()

        # Prefer returned "stories" and "epics" with keys
        items: List[Dict[str, Any]] = []
        if isinstance(data.get("stories"), list):
            items.extend(data["stories"])
        if isinstance(data.get("epics"), dict):
            # “epics” sometimes returned as {summary: KEY}
            for name, key in data["epics"].items():
                items.append({"key": key, "summary": name, "type": "Epic"})

        # Fallback to any generic fields your service may return
        for k in ("backlog", "issues", "items"):
            v = data.get(k)
            if isinstance(v, list):
                items.extend(v)

        # Normalize + add URLs
        normalized = []
        for it in items:
            key = it.get("key")
            summary = it.get("summary") or it.get("title") or ""
            issue_type = it.get("type") or it.get("issueType")
            url = f"{jira_base}/browse/{key}" if jira_base and key else None
            normalized.append({"key": key, "summary": summary, "type": issue_type, "url": url})

        top_n = int(os.getenv("JIRA_TOP_LIMIT", "10"))
        state["backlog"] = normalized[:top_n]

        # Human summary with links
        state.setdefault("summary_md", "")
        if state["backlog"]:
            lines = []
            for i in state["backlog"][:8]:
                if i.get("url") and i.get("key"):
                    lines.append(f"- [{i['key']}]({i['url']}) — {i.get('summary','')}")
                elif i.get("key"):
                    lines.append(f"- {i['key']} — {i.get('summary','')}")
                else:
                    lines.append(f"- {i.get('summary','')}")
            state["summary_md"] += "### Jira: generated backlog\n" + "\n".join(lines) + "\n\n"
        else:
            state["summary_md"] += "### Jira: generated backlog\n_no items returned_\n\n"

    except Exception as e:
        state.setdefault("errors", []).append(f"jira: {e}")
        state.setdefault("summary_md", "")
        state["summary_md"] += "### Jira: failed (see errors)\n\n"
        state["backlog"] = state.get("backlog", [])
    return state

# ===================== Graph =====================
graph = StateGraph(OrchestratorState)
graph.add_node("router", node_router)
graph.add_node("dispatch", node_dispatch)

graph.set_entry_point("router")
graph.add_edge("router", "dispatch")
graph.add_edge("dispatch", END)

APP = graph.compile()  # stateless is fine for now


# ===================== CLI =====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", help="Natural-language request from the PM", default="Plan next sprint from latest problems.")
    parser.add_argument("--apply", action="store_true", help="Force apply (overrides router's apply_changes=false)")
    # Back-compat convenience flags (optional)
    parser.add_argument("--project", default=None)
    parser.add_argument("--raw", default=None)
    parser.add_argument("--team", default=None)
    args = parser.parse_args()

    init: OrchestratorState = {
        "user_text": args.ask,
        # Allow CLI to force apply; otherwise router decides
        "auto_apply": bool(args.apply),
    }
    if args.project: init["project_key"] = args.project
    if args.team:    init["team_context"] = args.team
    if args.raw:     init["raw_input"] = args.raw

    out = APP.invoke(init)

    print("\n=== PLAN ===")
    print(json.dumps(out.get("router_plan"), indent=2))
    print("\n=== SUMMARY ===")
    print(out.get("summary_md"))
    print("\n=== LINKS ===")
    print(json.dumps({"pr_url": out.get("pr_url")}, indent=2))
    print("\n=== ERRORS ===")
    print(json.dumps(out.get("errors"), indent=2))
