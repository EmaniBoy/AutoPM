# to run: python -m agents.orchestrator --ask

#python api_server.py

# ai-agents/agents/orchestrator.py

# Extract key findings from the customer feedback about the digital banking app. Return ONLY a JSON array of strings.
# turn them into INVEST epics/stories for PB; plan a 3-week sprint (capacity 18); generate/update CI; Apply


from __future__ import annotations
import glob, math
from pathlib import Path
import numpy as np

import os, json, requests
from typing import TypedDict, Dict, Any, List, Optional

from dotenv import load_dotenv, find_dotenv

def _read_text_files(paths: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            for fp in glob.glob(os.path.join(p, "**", "*.*"), recursive=True):
                if fp.lower().endswith((".txt", ".md")):
                    try:
                        out[fp] = Path(fp).read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        pass
        elif os.path.isfile(p) and p.lower().endswith((".txt", ".md")):
            try:
                out[p] = Path(p).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return out

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = np.linalg.norm(a); db = np.linalg.norm(b)
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(np.dot(a, b) / (da * db))

def _format_legend(legend: List[Dict[str, Any]]) -> str:
    # [{"source": "...", "score": 0.82}, ...] -> "1) fileA (0.82), 2) fileB (0.75)"
    parts = []
    for i, row in enumerate(legend, 1):
        s = f"{i}) {os.path.basename(row['source'])}"
        if "score" in row:
            s += f" ({row['score']:.2f})"
        parts.append(s)
    return ", ".join(parts)


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
        # state["sources"] = inputs.get("sources", state.get("sources", []))
        # --- sources: normalize to list ---
        src = inputs.get("sources", state.get("sources", []))
        if isinstance(src, str):
            # accept comma-separated or single token
            src_list = [s.strip() for s in src.split(",") if s.strip()]
        elif isinstance(src, (list, tuple, set)):
            src_list = list(src)
        else:
            src_list = []
        state["sources"] = src_list
    except Exception as e:
        state.setdefault("errors", []).append(f"router: {e}")
        state["router_plan"] = {"intent": "other", "inputs": {}, "apply_changes": False}
    return state


def _run_research(state: OrchestratorState) -> None:
    """
    RAG-backed RESEARCH agent.
    Uses NemotronDocumentAnalyzer (like main.py) when vector DB is available.
    Otherwise, falls back to TF-IDF retrieval over ./ai-agents/data/*.
    """
    try:
        from pathlib import Path
        import numpy as np
        import json as _json
        import os

        user_query = state.get("raw_input") or state.get("user_text") or "Summarize problems and propose ideas."
        
        # ---- try to use NemotronDocumentAnalyzer (like main.py) ----
        index_path = os.path.abspath(os.path.join(_THIS_DIR, "..", "storage", "local_vectors.json"))
        use_customer_agent = False
        
        if os.path.isfile(index_path):
            try:
                from agents.customerAgent import NemotronDocumentAnalyzer, _infer_tasks_from_prompt
                from storage.vector_db import LocalVectorDB
                
                # Check if vector DB has embeddings
                raw = Path(index_path).read_text(encoding="utf-8")
                j = _json.loads(raw)
                vecs = j.get("vectors") if isinstance(j, dict) else (j if isinstance(j, list) else [])
                if vecs and isinstance(vecs[0], dict) and isinstance(vecs[0].get("embedding"), list) and vecs[0]["embedding"]:
                    # Initialize like main.py does
                    API_KEY = os.getenv("NVIDIA_API_KEYX")
                    EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODELX")
                    LLM_MODEL = os.getenv("NVIDIA_LLM_MODEX", "nvidia/nvidia-nemotron-nano-9b-v2")
                    DB_PATH = os.getenv("VECTOR_DB_PATH", "storage/local_vectors.json")
                    
                    if API_KEY and EMBED_MODEL:
                        retriever = LocalVectorDB(api_key=API_KEY, embed_model=EMBED_MODEL, db_path=DB_PATH)
                        agent = NemotronDocumentAnalyzer(api_key=API_KEY, model=LLM_MODEL, vector_db=retriever)
                        tasks = _infer_tasks_from_prompt(user_query)
                        
                        # Use the same approach as main.py
                        insight = agent.analyze_with_rag(
                            query=user_query,
                            document_type="customer_feedback",
                            top_k=5,
                            tasks=tasks,
                            allow_external=False,
                        )
                        
                        # Export to JSON like main.py
                        results_dir = os.path.abspath(os.path.join(_THIS_DIR, "..", "data", "results"))
                        os.makedirs(results_dir, exist_ok=True)
                        agent.export_to_json(insight, os.path.join(results_dir, "rag_analysis.json"))
                        
                        # Build summary from insight (like main.py output)
                        summary_parts = []
                        if tasks.summary and insight.summary:
                            summary_parts.append(f"📋 Summary:\n{insight.summary}")
                        if tasks.findings and insight.key_findings:
                            findings_text = "\n".join([f"   {i+1}. {f}" for i, f in enumerate(insight.key_findings)])
                            summary_parts.append(f"🔎 Key Findings:\n{findings_text}")
                        if tasks.problems and insight.problems_identified:
                            problems_text = "\n".join([f"   {i+1}. [{p.get('severity','?')}/{p.get('impact_area','?')}] {p.get('problem','')}" 
                                                       for i, p in enumerate(insight.problems_identified)])
                            summary_parts.append(f"⚠️ Problems:\n{problems_text}")
                        if tasks.ideas and insight.product_ideas:
                            ideas_text = "\n".join([f"   {i+1}. {idea.get('title','')} — {idea.get('impact','')}" 
                                                   for i, idea in enumerate(insight.product_ideas)])
                            summary_parts.append(f"💡 Ideas:\n{ideas_text}")
                        if tasks.metrics and insight.metrics:
                            metrics_text = "\n".join([f"   • {k}: {v}" for k, v in insight.metrics.items()])
                            summary_parts.append(f"📊 Metrics:\n{metrics_text}")
                            
                            # Generate metrics chart if metrics exist
                            try:
                                viz_path = os.path.join(results_dir, "metrics_chart.png")
                                agent.visualize_metrics(insight, save_path=viz_path)
                            except Exception:
                                pass
                        
                        summary = "\n\n".join(summary_parts) if summary_parts else "No results found."
                        
                        # Build backlog from findings/problems/ideas
                        backlog_items = []
                        if insight.key_findings:
                            backlog_items.extend([{"summary": f} for f in insight.key_findings])
                        if insight.problems_identified:
                            backlog_items.extend([{"summary": p.get('problem', '')} for p in insight.problems_identified])
                        if insight.product_ideas:
                            backlog_items.extend([{"summary": f"{idea.get('title', '')} — {idea.get('impact', '')}"} 
                                                  for idea in insight.product_ideas])
                        
                        if backlog_items:
                            state["backlog"] = backlog_items[:8]
                        
                        state.setdefault("summary_md", "")
                        state["summary_md"] += "### Research Findings\n" + summary + "\n\n"
                        
                        # Get sources from insight if available
                        if hasattr(insight, 'retrieved_sources') and insight.retrieved_sources:
                            # Format sources: retrieved_sources has doc_id, chunk_id, text_preview
                            sources_text = ", ".join([f"{os.path.basename(s.get('doc_id', 'unknown'))}" 
                                                     for s in insight.retrieved_sources[:5]])
                            state["summary_md"] += f"*Sources used:* {sources_text}\n\n"
                        
                        use_customer_agent = True
            except Exception as e:
                print(f"⚠️  Could not use NemotronDocumentAnalyzer: {e}")
                use_customer_agent = False
        
        # Fallback to original approach if customer agent not used
        if not use_customer_agent:
            # ---- resolve source folders ----
            def _resolve_sources(s):
                if isinstance(s, str):
                    s = [s]
                elif not isinstance(s, list):
                    s = []
                data_root = os.path.abspath(os.path.join(_THIS_DIR, "..", "data"))
                out = []
                for p in s:
                    if not os.path.isabs(p) and not p.startswith("."):
                        p = os.path.join(data_root, p)
                    out.append(os.path.abspath(p))
                return out

            default_dir = os.path.abspath(os.path.join(_THIS_DIR, "..", "data", "research"))
            banking_dir = os.path.abspath(os.path.join(_THIS_DIR, "..", "data", "banking_corpus"))
            
            # Resolve sources and filter out non-existent directories
            resolved = _resolve_sources(state.get("sources"))
            source_paths = [p for p in resolved if os.path.isdir(p)]
            
            # If no valid sources found, use fallback directories
            if not source_paths:
                source_paths = [p for p in [default_dir, banking_dir] if os.path.isdir(p)]

            # ---- try vector DB iff it has embeddings ----
            use_vectors, vecs = False, []
            if os.path.isfile(index_path):
                raw = Path(index_path).read_text(encoding="utf-8")
                j = _json.loads(raw)
                vecs = j.get("vectors") if isinstance(j, dict) else (j if isinstance(j, list) else [])
                if vecs and isinstance(vecs[0], dict) and isinstance(vecs[0].get("embedding"), list) and vecs[0]["embedding"]:
                    use_vectors = True  # only if we truly have embeddings

            retrieved, legend = [], []

            if use_vectors:
                # ---- vector mode (cosine to weighted centroid) ----
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    texts = [(v.get("text") or "") for v in vecs]
                    tfidf = TfidfVectorizer(max_features=4096).fit(texts + [user_query])
                    q_vec_lex = tfidf.transform([user_query]).toarray()[0]
                    weights = tfidf.transform(texts).toarray() @ q_vec_lex
                except Exception:
                    weights = np.ones(len(vecs), dtype=float)

                M = max(5, min(50, len(vecs)))
                idx_sorted = np.argsort(weights)[::-1][:M]

                emb_dim = len(vecs[0]["embedding"])
                centroid = np.zeros((emb_dim,), dtype=float)
                total_w = 0.0
                for i in idx_sorted:
                    e = np.array(vecs[i]["embedding"], dtype=float)
                    w = float(weights[i] + 1e-6)
                    centroid += w * e
                    total_w += w
                if total_w > 0:
                    centroid /= total_w

                def _cos(a, b):
                    na = np.linalg.norm(a); nb = np.linalg.norm(b)
                    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

                scores = []
                for v in vecs:
                    e = np.array(v.get("embedding") or [], dtype=float)
                    scores.append(_cos(centroid, e) if e.size else 0.0)

                order = np.argsort(np.array(scores))[::-1][:8]
                for i in order:
                    v = vecs[int(i)]
                    retrieved.append({"source": v.get("source") or v.get("id") or "unknown", "text": v.get("text", "")})
                    legend.append({"source": v.get("source") or "unknown", "score": float(scores[int(i)])})

            else:
                # ---- TF-IDF fallback over files ----
                from pathlib import Path
                def _read_text_files(paths):
                    out = {}
                    for p in paths:
                        if os.path.isdir(p):
                            for root, _, files in os.walk(p):
                                for fn in files:
                                    if fn.lower().endswith((".txt", ".md")):
                                        fp = os.path.join(root, fn)
                                        try:
                                            out[fp] = Path(fp).read_text(encoding="utf-8", errors="ignore")
                                        except Exception:
                                            pass
                        elif os.path.isfile(p):
                            try:
                                out[p] = Path(p).read_text(encoding="utf-8", errors="ignore")
                            except Exception:
                                pass
                    return out

                corpus = _read_text_files(source_paths) if source_paths else {}
                if not corpus and not use_vectors:
                    # Only raise error if we're not using vectors
                    raise RuntimeError(f"No research files found. Checked: {source_paths or [default_dir, banking_dir]}")

                files, texts = list(corpus.keys()), list(corpus.values())
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity
                    vec = TfidfVectorizer(max_features=8192)
                    X = vec.fit_transform(texts)
                    q = vec.transform([user_query])
                    sims = cosine_similarity(q, X)[0]
                except Exception:
                    # ultra-safe lexical fallback
                    sims = np.array([len(t) for t in texts], dtype=float)

                topk = np.argsort(sims)[::-1][:8]
                for idx in topk:
                    retrieved.append({"source": files[idx], "text": texts[idx]})
                    legend.append({"source": files[idx], "score": float(sims[idx])})

            # ---- build grounded prompt (only if not using customer agent) ----
            if not use_customer_agent:
                blocks = []
                for i, ch in enumerate(retrieved, 1):
                    snip = (ch["text"] or "").strip()
                    if len(snip) > 1200: snip = snip[:1200] + " ..."
                    blocks.append(f"[{i}] Source: {os.path.basename(ch['source'])}\n{snip}")
                ctx = "\n\n".join(blocks) if blocks else "(no external context)"

                system_prompt = (
                    "You are a product research analyst. Using ONLY the provided context, do the following:\n"
                    "1) List 5 concrete problems/opportunities.\n"
                    "2) Propose 3 crisp product ideas.\n"
                    "Rules: Return markdown bullets, each starting with '- '. No preamble. Be concise."
                )
                user_msg = f"User query: {user_query}\n\nGrounded context (top {len(retrieved)} chunks):\n{ctx}"

                summary = nv_chat(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_msg}],
                    max_tokens=700,
                )

                items = [{"summary": line.strip("- ").strip()} for line in summary.splitlines() if line.strip().startswith("-")][:8]
                if items:
                    state["backlog"] = items

                def _legend_md(legend_list):
                    if not legend_list: return ""
                    parts = [f"{os.path.basename(x.get('source','unknown'))} ({x.get('score',0):.3f})" for x in legend_list[:5]]
                    return ", ".join(parts)

                state.setdefault("summary_md", "")
                state["summary_md"] += "### Research Findings\n" + summary.strip() + "\n\n"
                lg = _legend_md(legend)
                if lg:
                    state["summary_md"] += f"*Sources used:* {lg}\n\n"

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
