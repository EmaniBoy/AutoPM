#python -m agents.devops_planner --apply

import base64
import json
import os
import re
import uuid
from typing import Dict, Any
import requests  # required for branch creation + PR comment

from agents.shared import (
    nv_chat,
    gh_get,
    gh_put_file,
    gh_open_pr,
    GITHUB_REPO,
    GITHUB_DEFAULT_BRANCH,
)

# ---------- Prompt tuned for mini-4B (JSON-only, concise) ----------
SYSTEM_PROMPT = """You are a DevOps Planner.
Return ONLY a single valid JSON object (no backticks, no prose) with keys:
- "filename": string (e.g., "ci.yml")
- "yaml": string (a complete GitHub Actions workflow with jobs: lint, test, build, and release on tags if applicable)
- "sprint": object with "weeks": [ { "goals":[], "tasks":[], "capacity":int, "risks":[] } ] (exactly 3 items)

Assumptions:
- Use only free GitHub-hosted runners.
- Prefer the simplest steps:
  - Python: use "pip install -r requirements.txt" if present, run "pytest" if tests exist.
  - Node: use "npm ci", "npm test", "npm run build" if they exist.
  - Java: use Maven (pom.xml) or Gradle (build.gradle) as detected.
- Keep YAML minimal and fast; avoid matrix unless necessary.
- Keep sprint bullets concise; realistic but short.
"""

CORRECTIVE_PROMPT = """Your previous output did not contain a valid GitHub Actions YAML (missing 'on:' or 'jobs:').
Return ONLY JSON with keys filename, yaml, sprint. Ensure yaml has both 'on:' and 'jobs:'.
Keep it minimal and valid.
"""

# ---------- Helpers ----------

def _extract_json(s: str) -> str:
    """Strip code fences and return the largest {...} block as a fallback."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    m = re.search(r"\{.*\}\s*$", s, flags=re.DOTALL)
    return m.group(0) if m else s

def _yaml_min_sanity(yaml_text: str) -> None:
    """
    Minimal sanity checks without adding PyYAML:
    - must contain 'on:' and 'jobs:'
    """
    txt = yaml_text.lower()
    if "on:" not in txt or "jobs:" not in txt:
        raise ValueError("Generated YAML seems incomplete (missing 'on:' or 'jobs:').")

def _ensure_base_branch(repo: str, base_branch: str) -> None:
    """
    Make sure base_branch exists. If the repo is empty (no ref), create an initial commit.
    """
    ref = gh_get(f"/repos/{repo}/git/ref/heads/{base_branch}")
    if ref is None:
        _init_repo_with_readme(repo, base_branch)

def _init_repo_with_readme(repo: str, base_branch: str) -> None:
    """
    Initialize an empty repository by creating README.md on the desired base branch.
    """
    content_b64 = base64.b64encode(b"# Project\n\nInitialized by DevOps Planner.\n").decode("utf-8")
    gh_put_file(
        repo,
        "README.md",
        content_b64,
        message=f"chore: initialize {base_branch}",
        branch=base_branch,
        sha=None,
    )

def _gh_create_branch(repo: str, base_branch: str, new_branch: str) -> None:
    """
    Create a branch ref from base_branch. If base_branch doesn't exist (empty repo),
    initialize it first. Ignore 422 if the new ref already exists.
    """
    _ensure_base_branch(repo, base_branch)
    ref = gh_get(f"/repos/{repo}/git/ref/heads/{base_branch}")
    if not ref:
        raise RuntimeError(f"Failed to create/find base branch '{base_branch}'")
    base_sha = ref["object"]["sha"]
    url = f"https://api.github.com/repos/{repo}/git/refs"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
                 "Accept": "application/vnd.github+json"},
        json={"ref": f"refs/heads/{new_branch}", "sha": base_sha},
        timeout=60,
    )
    if r.status_code not in (201, 422):
        r.raise_for_status()

def _gh_comment_issue(repo: str, issue_number: int, body: str) -> None:
    """Post a comment to a PR (issues API)."""
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
                 "Accept": "application/vnd.github+json"},
        json={"body": body},
        timeout=60,
    )
    r.raise_for_status()

def _normalize_sprint(sprint_in: Any) -> Dict[str, Any]:
    """
    Accepts:
      - {"weeks":[...]}
      - [...]  (list of weeks)
      - None
    Returns: {"weeks":[...]} with exactly a list.
    """
    if isinstance(sprint_in, dict) and "weeks" in sprint_in and isinstance(sprint_in["weeks"], list):
        return sprint_in
    if isinstance(sprint_in, list):
        return {"weeks": sprint_in}
    # default empty
    return {"weeks": []}

def format_sprint_markdown(sprint: Any) -> str:
    sprint = _normalize_sprint(sprint)
    lines = ["# 3-Week Sprint Plan"]
    weeks = sprint.get("weeks", [])
    for i, wk in enumerate(weeks, start=1):
        lines += [
            f"\n## Week {i}",
            "**Goals**",
        ]
        for g in wk.get("goals", []):
            lines.append(f"- {g}")
        lines += [
            "\n**Tasks**",
        ]
        for t in wk.get("tasks", []):
            lines.append(f"- {t}")
        lines += [
            f"\n**Capacity**: {wk.get('capacity', 0)} pts",
            "\n**Risks**",
        ]
        for r in wk.get("risks", []):
            lines.append(f"- {r}")
    return "\n".join(lines)

# ---------- Fallback synthesis (no LLM) ----------

def _synthesize_yaml(repo_summary: Dict[str, Any]) -> str:
    """Produce a safe minimal workflow given detected signals."""
    files = repo_summary.get("files_detected", {})
    has_py = files.get("python")
    has_node = files.get("node")
    has_pom = files.get("maven")
    has_gradle = files.get("gradle")
    has_tests = files.get("tests_present")

    header = (
        "name: CI\n\n"
        "on:\n"
        "  push:\n"
        "    branches: [ main ]\n"
        "  pull_request:\n"
        "    branches: [ main ]\n\n"
        "jobs:\n"
    )

    def block_python():
        return (
            "  lint:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-python@v5\n"
            "        with:\n"
            "          python-version: '3.x'\n"
            "      - run: |\n"
            "          pip install flake8 || true\n"
            "          flake8 . || true\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    needs: lint\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-python@v5\n"
            "        with:\n"
            "          python-version: '3.x'\n"
            "      - run: |\n"
            "          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi\n"
            "          pip install pytest || true\n"
            "          if [ -d tests ] || ls -1 *_test.py 2>/dev/null | grep -q .; then pytest -q; else echo 'No tests'; fi\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    needs: test\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - run: echo 'Build placeholder'\n"
            "  release:\n"
            "    if: startsWith(github.ref, 'refs/tags/')\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: softprops/action-gh-release@v2\n"
            "        with:\n"
            "          draft: true\n"
        )

    def block_node():
        return (
            "  lint:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-node@v4\n"
            "        with:\n"
            "          node-version: '20'\n"
            "      - run: |\n"
            "          npm ci || true\n"
            "          npx eslint . || true\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    needs: lint\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-node@v4\n"
            "        with:\n"
            "          node-version: '20'\n"
            "      - run: |\n"
            "          npm ci || true\n"
            "          if npm run | grep -q ' test'; then npm test --silent || true; else echo 'No tests'; fi\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    needs: test\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-node@v4\n"
            "        with:\n"
            "          node-version: '20'\n"
            "      - run: |\n"
            "          npm ci || true\n"
            "          if npm run | grep -q ' build'; then npm run build; else echo 'No build'; fi\n"
            "  release:\n"
            "    if: startsWith(github.ref, 'refs/tags/')\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: softprops/action-gh-release@v2\n"
            "        with:\n"
            "          draft: true\n"
        )

    def block_java():
        # Use Maven if pom.xml detected, else Gradle if build.gradle
        tool = "mvn -B -q" if has_pom else "gradle -q"
        return (
            "  build-test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - name: Set up JDK\n"
            "        uses: actions/setup-java@v4\n"
            "        with:\n"
            "          distribution: 'temurin'\n"
            "          java-version: '17'\n"
            "      - run: |\n"
            f"          {tool} test || true\n"
            "      - run: |\n"
            f"          {tool} package || echo 'Build placeholder'\n"
            "  release:\n"
            "    if: startsWith(github.ref, 'refs/tags/')\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: softprops/action-gh-release@v2\n"
            "        with:\n"
            "          draft: true\n"
        )

    def block_empty():
        return (
            "  lint:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - run: echo 'No code yet — add files to enable real checks.'\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    needs: lint\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - run: echo 'No tests yet.'\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    needs: test\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - run: echo 'Build placeholder.'\n"
            "  release:\n"
            "    if: startsWith(github.ref, 'refs/tags/')\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: softprops/action-gh-release@v2\n"
            "        with:\n"
            "          draft: true\n"
        )

    if has_py:
        return header + block_python()
    if has_node:
        return header + block_node()
    if has_pom or has_gradle:
        return header + block_java()
    return header + block_empty()

# ---------- Core agent steps ----------

def detect_repo_state() -> Dict[str, Any]:
    """Read basic signals from repo to tailor the pipeline."""
    repo = GITHUB_REPO
    branch = GITHUB_DEFAULT_BRANCH

    # languages
    langs = gh_get(f"/repos/{repo}/languages") or {}

    # file inventory
    tree = gh_get(f"/repos/{repo}/git/trees/{branch}?recursive=1") or {"tree": []}
    files = [t["path"] for t in tree.get("tree", []) if t.get("type") == "blob"]

    has_py = any(p.endswith(".py") for p in files)
    has_node = any(p.endswith("package.json") for p in files)
    has_pom = any(p.endswith("pom.xml") for p in files)
    has_gradle = any("build.gradle" in p for p in files)
    has_tests = any(
        ("tests/" in p) or p.endswith("_test.py") or p.endswith(".spec.ts") or p.endswith(".test.ts")
        for p in files
    )
    has_docker = any(p.lower() == "dockerfile" or p.endswith("/dockerfile") for p in files)
    has_workflows = any(p.startswith(".github/workflows/") for p in files)

    summary = {
        "languages": langs,
        "files_detected": {
            "python": has_py,
            "node": has_node,
            "maven": has_pom,
            "gradle": has_gradle,
            "tests_present": has_tests,
            "dockerfile": has_docker,
            "existing_workflows": has_workflows,
        },
        "examples": [p for p in files if p.startswith(".github/workflows/")][:5],
    }
    return summary

def _ask_model(repo_summary: Dict[str, Any], corrective: bool = False) -> Dict[str, Any]:
    compact = json.dumps(repo_summary, separators=(",", ":"), ensure_ascii=False)
    user = f"Repository summary:\n{compact[:5000]}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if corrective:
        messages.append({"role": "system", "content": CORRECTIVE_PROMPT})
    messages.append({"role": "user", "content": user})

    out = nv_chat(messages, max_tokens=600, temperature=0.1)
    try:
        return json.loads(out)
    except Exception:
        return json.loads(_extract_json(out))

def propose_plan(repo_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use the model, retry with a corrective hint if needed, then synthesize fallback YAML if still invalid.
    """
    plan = _ask_model(repo_summary, corrective=False)

    def valid_yaml(p):
        y = p.get("yaml")
        return isinstance(y, str) and ("on:" in y.lower()) and ("jobs:" in y.lower())

    if not plan.get("filename") or not valid_yaml(plan):
        # Retry once with corrective system hint
        plan = _ask_model(repo_summary, corrective=True)

    if not plan.get("filename") or not valid_yaml(plan):
        # Final fallback: synthesize YAML locally
        synthesized_yaml = _synthesize_yaml(repo_summary)
        plan = {
            "filename": "ci.yml",
            "yaml": synthesized_yaml,
            "sprint": plan.get("sprint") or {
                "weeks": [
                    {"goals": ["Bootstrap repo and CI"], "tasks": ["Add basic code skeleton"], "capacity": 8, "risks": ["New repo, unknowns"]},
                    {"goals": ["Add tests & lint rules"], "tasks": ["Set up pytest/eslint"], "capacity": 10, "risks": ["Tooling overhead"]},
                    {"goals": ["Harden CI & release"], "tasks": ["Tag-based release flow"], "capacity": 10, "risks": ["Release permissions"]},
                ]
            },
        }

    plan["sprint"] = _normalize_sprint(plan.get("sprint"))

    _yaml_min_sanity(plan["yaml"])
    return plan

def upsert_workflow(filename: str, yaml_text: str, base_branch: str, sprint_md: str | None = None) -> Dict[str, Any]:
    """
    Create or update .github/workflows/<filename> on a temporary branch,
    open a PR, and (optionally) post the sprint plan as a PR comment.
    """
    path = f".github/workflows/{filename}"
    tmp_branch = f"devops-planner-{uuid.uuid4().hex[:8]}"

    # create branch from base
    _gh_create_branch(GITHUB_REPO, base_branch, tmp_branch)

    # does file exist on tmp branch?
    existing = gh_get(f"/repos/{GITHUB_REPO}/contents/{path}?ref={tmp_branch}")
    sha = existing.get("sha") if existing else None

    content_b64 = base64.b64encode(yaml_text.encode("utf-8")).decode("utf-8")
    res = gh_put_file(
        GITHUB_REPO,
        path,
        content_b64,
        message=f"Add/update {filename} via DevOps Planner",
        branch=tmp_branch,
        sha=sha,
    )

    pr = gh_open_pr(
        GITHUB_REPO,
        head=tmp_branch,
        base=base_branch,
        title=f"DevOps Planner: add {filename}",
        body="This PR was generated by the DevOps Planner agent.",
    )

    # Optional: post sprint plan as a PR comment (nice UX for reviewers)
    if sprint_md:
        try:
            _gh_comment_issue(GITHUB_REPO, pr["number"], sprint_md)
        except Exception as e:
            print(f"Warning: failed to post sprint plan comment: {e}")

    return {"branch": tmp_branch, "commit": res.get("commit", {}), "pr": pr}

def run(dry_run: bool = True, create_pr: bool = False):
    repo_state = detect_repo_state()
    plan = propose_plan(repo_state)
    filename = (plan["filename"] or "ci.yml").strip()
    yaml_text = plan["yaml"]
    sprint_md = format_sprint_markdown(plan.get("sprint", {}))

    print("=== Proposed Workflow File ===")
    print(f".github/workflows/{filename}\n")
    print(yaml_text)
    print("\n=== Sprint Plan (Markdown) ===")
    print(sprint_md)

    if not dry_run and create_pr:
        result = upsert_workflow(filename, yaml_text, GITHUB_DEFAULT_BRANCH, sprint_md)
        print("\nOpened PR:", result["pr"]["html_url"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Create branch, commit workflow, open PR, comment sprint plan.")
    args = parser.parse_args()
    run(dry_run=not args.apply, create_pr=args.apply)
