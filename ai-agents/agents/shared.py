import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

# --- NVIDIA (Nemotron-Mini-4B-Instruct via OpenAI-compatible endpoint) ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "nvidia/nemotron-mini-4b-instruct")

def nv_chat(messages, temperature=0.2, max_tokens=700, tools=None, tool_choice="auto", stream=False):
    """
    Thin wrapper over NVIDIA's OpenAI-compatible /v1/chat/completions.
    Works with Nemotron-Mini-4B-Instruct via build.nvidia.com.
    """
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]

# --- GitHub helpers ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_DEFAULT_BRANCH = os.getenv("GITHUB_DEFAULT_BRANCH", "main")

def _gh_headers():
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is not set")
    return {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

def gh_get(path, accept="application/vnd.github+json"):
    headers = _gh_headers()
    headers["Accept"] = accept
    r = requests.get(f"https://api.github.com{path}", headers=headers, timeout=60)
    # New: treat empty-repo (409) and not-found (404) as "no data"
    if r.status_code in (404, 409):
        return None
    r.raise_for_status()
    return r.json()

def gh_put_file(repo, path, content_b64, message, branch, sha=None):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    payload = {"message": message, "content": content_b64, "branch": branch}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_gh_headers(), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def gh_open_pr(repo, head, base, title, body=""):
    url = f"https://api.github.com/repos/{repo}/pulls"
    payload = {"title": title, "head": head, "base": base, "body": body}
    r = requests.post(url, headers=_gh_headers(), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()
