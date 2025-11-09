# diag_nvidia.py

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from ai-agents directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

API_BASE = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("NVIDIA_API_KEYX")
PRIMARY = os.getenv("NVIDIA_LLM_MODEX", "nvidia/nvidia-nemotron-nano-9b-v2")
SECOND = os.getenv("SECONDARY_LLM_MODEL", "meta/llama-3.1-8b-instruct")

if not API_KEY:
    print("Missing NVIDIA_API_KEYX in .env")
    sys.exit(1)

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

prompt = 'Reply with exactly this JSON array: ["ok"]. Nothing else.'

# Use STRING content, not array-form
messages_string_content = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": prompt},
]

def call(model):
    return client.chat.completions.create(
        model=model,
        messages=messages_string_content,  # <-- string content form
        temperature=0.0,
        max_tokens=32,
        top_p=0.95,
        stream=False,
    )

def extract_text(resp):
    # Try standard place
    msg = resp.choices[0].message if resp and resp.choices else None
    content = (msg.content or "").strip() if msg else ""
    
    # Check reasoning_content (Nemotron thinking mode)
    reasoning = getattr(msg, "reasoning_content", None) or ""
    if reasoning:
        content = reasoning.strip()
    
    # Some SDKs place structured/json outputs here
    parsed = getattr(msg, "parsed", None)
    # Tool calls?
    tools = getattr(msg, "tool_calls", None)
    
    finish_reason = resp.choices[0].finish_reason if resp and resp.choices else None
    
    return content, parsed, tools, finish_reason

def dump(resp, path):
    Path("logs").mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        try:
            obj = resp.model_dump()
        except Exception:
            obj = json.loads(resp.json()) if hasattr(resp, "json") else json.loads(str(resp))
        json.dump(obj, f, indent=2)

for i, model in enumerate([PRIMARY, SECOND], start=1):
    print(f"\n--- Attempt {i}: {model} ---")
    try:
        r = call(model)
        dump(r, f"logs/diag_last_{i}.json")
        content, parsed, tools, finish_reason = extract_text(r)
        print("content:", repr(content))
        print("parsed :", parsed if parsed is not None else "None")
        print("tools  :", tools if tools else "None")
        print("finish_reason:", finish_reason)
        if content:
            print("\n✅ SUCCESS:", content)
            break
        elif parsed:
            print("\n✅ SUCCESS (parsed):", json.dumps(parsed))
            break
        else:
            print("⚠️ No text content. Check logs/diag_last_*.json to see raw fields.")
    except Exception as e:
        print("❌ Error:", e)
