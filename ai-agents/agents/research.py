# agents/research.py
import os
import json

def fetch_research_snippets(query: str, max_snippets: int = 3) -> list:
    """
    Optional external research (requires TAVILY_API_KEY or SERPAPI_KEY).
    Returns short snippets to enrich context. If no key, returns [].
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            import requests
            r = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "max_results": max_snippets},
                timeout=5
            )
            r.raise_for_status()
            data = r.json()
            out = []
            for item in data.get("results", [])[:max_snippets]:
                snippet = item.get("content") or item.get("snippet") or ""
                if snippet:
                    out.append(snippet.strip())
            return out
        except Exception as e:
            print(f"⚠️  Research fetch failed: {e}")
            return []
    
    # (Optional) add other providers here (SERPAPI, etc.)
    
    return []