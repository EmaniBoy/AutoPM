# auth, tools, utils

import os, requests
from dotenv import load_dotenv
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "nvidia/nemotron-4-340b-instruct")

def nv_chat(messages, temperature=0.2, max_tokens=800):
    """Minimal Nemotron-4 Instruct chat wrapper."""
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]

# --- GitHub helpers ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_DEFAULT_BRANCH = os.getenv("GITHUB_DEFAULT_BRANCH", "main")

def gh_get(path, accept="application/vnd.github+json"):
    r = requests.get(f"https://api.github.com{path}",
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": accept}, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def gh_put_file(repo, path, content_b64, message, branch, sha=None):
    import requests
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    payload = {"message": message, "content": content_b64, "branch": branch}
    if sha: payload["sha"] = sha
    r = requests.put(url,
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"},
        json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def gh_open_pr(repo, head, base, title, body=""):
    url = f"https://api.github.com/repos/{repo}/pulls"
    payload = {"title": title, "head": head, "base": base, "body": body}
    r = requests.post(url,
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"},
        json=payload, timeout=60)
    r.raise_for_status()
    return r.json()
