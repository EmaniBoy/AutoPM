# ai-agents/agents/shared.py
from __future__ import annotations
import os, base64, requests
from typing import Any, Dict, Optional
from dotenv import load_dotenv, find_dotenv

# -----------------------
# Load .env deterministically
# -----------------------
# 1) Try ai-agents/.env relative to this file
_THIS_DIR = os.path.dirname(__file__)
_AI_AGENTS_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_ENV_PATH = os.path.join(_AI_AGENTS_DIR, ".env")

# Load explicit path first (override so it wins), then fall back to default search chain.
if os.path.isfile(_ENV_PATH):
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
# Also try any parent .env in case you're running from a different CWD
_auto = find_dotenv(usecwd=True)
if _auto:
    load_dotenv(dotenv_path=_auto, override=False)

# -----------------------
# NVIDIA (Nemotron Mini 4B Instruct via OpenAI-compatible endpoint)
# -----------------------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "nvidia/nemotron-mini-4b-instruct")
NVIDIA_URL = os.getenv("NVIDIA_CHAT_URL", "https://integrate.api.nvidia.com/v1/chat/completions")

def nv_chat(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 700,
    tools: Optional[list[dict]] = None,
    tool_choice: str = "auto",
    stream: bool = False,
) -> str:
    """Thin wrapper over NVIDIA's OpenAI-compatible /v1/chat/completions."""
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set")
    if not CHAT_MODEL:
        raise RuntimeError("NVIDIA_CHAT_MODEL is not set")

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    r = requests.post(NVIDIA_URL, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    j = r.json()
    # Some models return tool calls; your current agents expect plain content.
    return j["choices"][0]["message"].get("content", "") or ""

# -----------------------
# GitHub helpers
# -----------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_DEFAULT_BRANCH = os.getenv("GITHUB_DEFAULT_BRANCH", "main")

def _gh_headers(accept: str = "application/vnd.github+json"):
    tok = os.getenv("GITHUB_TOKEN")  # re-read in case env changes at runtime
    if not tok:
        raise RuntimeError("GITHUB_TOKEN is not set")
    return {
        "Authorization": f"Bearer {tok}",
        "Accept": accept,
        # Recommended header (fine-grained tokens behave better with it)
        "X-GitHub-Api-Version": "2022-11-28",
    }

def gh_get(path: str, accept: str = "application/vnd.github+json"):
    """GET helper that returns None for 404/409 (empty repos, missing objects)."""
    r = requests.get(f"https://api.github.com{path}", headers=_gh_headers(accept), timeout=60)
    if r.status_code in (404, 409):
        return None
    r.raise_for_status()
    return r.json()

def gh_put_file(repo: str, path: str, content_b64: str, message: str, branch: str, sha: Optional[str] = None):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    payload = {"message": message, "content": content_b64, "branch": branch}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_gh_headers(), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def gh_open_pr(repo: str, head: str, base: str, title: str, body: str = ""):
    url = f"https://api.github.com/repos/{repo}/pulls"
    payload = {"title": title, "head": head, "base": base, "body": body}
    r = requests.post(url, headers=_gh_headers(), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def gh_comment_issue(repo: str, number: int, body: str):
    """Comment on an issue/PR (PRs are issues under the hood)."""
    url = f"https://api.github.com/repos/{repo}/issues/{number}/comments"
    r = requests.post(url, headers=_gh_headers(), json={"body": body}, timeout=60)
    r.raise_for_status()
    return r.json()

def gh_get_ref(repo: str, ref: str):
    """Get a git ref, e.g., 'heads/main' or 'tags/v1'."""
    r = requests.get(f"https://api.github.com/repos/{repo}/git/ref/{ref}", headers=_gh_headers(), timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def gh_create_ref(repo: str, ref: str, sha: str):
    """Create a ref: ref='refs/heads/branch-name'."""
    url = f"https://api.github.com/repos/{repo}/git/refs"
    r = requests.post(url, headers=_gh_headers(), json={"ref": ref, "sha": sha}, timeout=60)
    r.raise_for_status()
    return r.json()

# Convenience: base64 encode text for gh_put_file
def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")
