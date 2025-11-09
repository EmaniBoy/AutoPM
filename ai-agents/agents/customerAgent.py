import os
import json
import re
import time
from typing import Dict, List, Optional, Any
from openai import OpenAI
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from datetime import datetime
from agents.research import fetch_research_snippets

# PM System Prompt
PM_SYSTEM = """You are a senior Product Manager in banking/fintech.

Principles:
- Be precise and impact-focused (conversion, retention, NPS, compliance risk).
- Cite evidence from provided context; avoid speculation.
- Prefer JSON-only outputs when requested. No preamble, no markdown.

Heuristics:
- Severity: High if it impacts money movement, security/compliance, or >10% of active users, else Medium/Low.
- Impact area taxonomy: UX, Performance, Features, Support, Compliance, Enterprise.

When JSON is requested, return valid JSON only."""

# Few-shot examples
FEW_SHOT_FINDINGS = """Examples:
Input: "Users abandon transfers at shipping; iOS 17 crashes adding payees."
Output: ["Transfer flow overly long; abandonment rising", "iOS 17 crash adding payees"]

Now do the same for the document below.

"""

FEW_SHOT_PROBLEMS = """Examples:
Input: "App crashes on iOS 17 when adding payees. 15% of transfers abandoned at shipping step."
Output: [{"problem": "iOS 17 crash when adding payees", "severity": "High", "impact_area": "UX"}, {"problem": "Transfer abandonment at shipping step (15%)", "severity": "High", "impact_area": "UX"}]

Now do the same for the document below.

"""

FEW_SHOT_IDEAS = """Examples:
Input: Problems: ["iOS 17 crash when adding payees", "Transfer abandonment at shipping step"]
Output: [{"title": "Payee Management Fix", "description": "Fix iOS 17 crash and add payee validation", "impact": "Reduce support tickets by 40%"}, {"title": "Streamlined Transfer Flow", "description": "Reduce shipping step friction with auto-fill", "impact": "Reduce abandonment by 25%"}]

Now do the same for the problems below.

"""

# Canonical metrics mapping
CANONICAL_METRICS = {
    "abandon": "transfer_abandon_rate",
    "crash": "app_crash_rate",
    "retention": "new_user_retention_30d",
    "biometric": "biometric_failure_rate",
    "nps": "nps",
    "api backlog": "enterprise_api_backlog",
    "hold": "compliance_hold_incidents",
    "chargeback": "chargeback_rate",
    "kyc": "kyc_failure_rate",
}


def _extract_context_metrics(text: str) -> dict:
    """Extract metrics from context using regex patterns"""
    metrics = {}

    # Find % values with nearby keywords
    for m in re.finditer(r'(\b[\w\s]{0,25}\b)(\d+(?:\.\d+)?)\s*%', text, flags=re.I):
        window, val = m.group(1).lower(), float(m.group(2))
        key = None
        for k, canon in CANONICAL_METRICS.items():
            if k in window:
                key = canon
                break
        if key:
            metrics[key] = val / 100.0  # store as 0..1

    # Find integer-ish counts near "backlog/requests/tickets"
    for m in re.finditer(r'(backlog|requests|tickets)[\s\w]{0,15}(\d{1,5})', text, flags=re.I):
        key = "enterprise_api_backlog"
        metrics[key] = int(m.group(2))

    return metrics


@dataclass
class AnalysisTasks:
    summary: bool = False
    findings: bool = False
    problems: bool = False
    ideas: bool = False
    metrics: bool = False  # OFF by default now


def _infer_tasks_from_prompt(prompt: str) -> AnalysisTasks:
    p = (prompt or "").lower()
    # keyword sets
    k_summary = ("summary", "summarize", "tl;dr", "2-3 sentences", "4-5 sentences")
    k_findings = ("key findings", "findings", "insights", "highlights")
    k_problems = ("problems", "pain points", "issues", "risks", "bottlenecks")
    k_ideas = ("ideas", "solutions", "recommendations", "features", "product ideas", "roadmap")
    k_metrics = ("metrics", "kpis", "numbers", "rates", "counts", "statistics", "conversion", "retention", "crashes", "kyc", "chargebacks")

    wants = AnalysisTasks(
        summary=any(k in p for k in k_summary),
        findings=any(k in p for k in k_findings),
        problems=any(k in p for k in k_problems),
        ideas=any(k in p for k in k_ideas),
        metrics=any(k in p for k in k_metrics),
    )

    # "only/just X" semantics disable everything else
    # But first check if a keyword appears BEFORE "only/just" (e.g., "metrics only relevant to...")
    only_match = re.search(r"\b(only|just)\b", p)
    if only_match:
        only_pos = only_match.start()
        text_before_only = p[:only_pos]
        text_after_only = p[only_match.end():].strip()
        
        # Check if any keyword appears before "only/just"
        found_before = AnalysisTasks(
            summary=any(k in text_before_only for k in k_summary),
            findings=any(k in text_before_only for k in k_findings),
            problems=any(k in text_before_only for k in k_problems),
            ideas=any(k in text_before_only for k in k_ideas),
            metrics=any(k in text_before_only for k in k_metrics),
        )
        
        # If we found a keyword before "only", use that and disable others
        if any([found_before.summary, found_before.findings, found_before.problems, found_before.ideas, found_before.metrics]):
            wants = AnalysisTasks(
                summary=found_before.summary,
                findings=found_before.findings,
                problems=found_before.problems,
                ideas=found_before.ideas,
                metrics=found_before.metrics,
            )
        else:
            # No keyword before "only", check what comes after
            def hit(keys): return any(k in text_after_only for k in keys)
            
            wants = AnalysisTasks(
                summary=hit(k_summary),
                findings=hit(k_findings),
                problems=hit(k_problems),
                ideas=hit(k_ideas),
                metrics=hit(k_metrics),
            )

    return wants


@dataclass
class DocumentInsight:
    """Structure for document analysis insights"""
    summary: str
    key_findings: List[str]
    problems_identified: List[Dict[str, str]]
    product_ideas: List[Dict[str, str]]
    sentiment_analysis: Dict[str, Any]
    metrics: Dict[str, Any]
    timestamp: str
    # RAG-specific fields
    retrieved_sources: Optional[List[Dict[str, str]]] = None
    query_used: Optional[str] = None
    rag_enabled: bool = False


def _is_txt_doc(doc_id: str) -> bool:
    return doc_id.lower().endswith(".txt") or ".txt#" in doc_id.lower()


def _prefer_txt_first(chunks: List[Dict]) -> List[Dict]:
    txt = [c for c in chunks if _is_txt_doc(c.get("doc_id", ""))]
    other = [c for c in chunks if c not in txt]
    return txt + other


class NemotronDocumentAnalyzer:
    """
    AI Agent for analyzing customer data and documents using NVIDIA Nemotron LLM
    Optimized for nvidia-nemotron-nano-9b-v2
    Supports both direct document analysis and RAG-based analysis
    """

    def __init__(self, api_key: str, model: str = "nvidia/nvidia-nemotron-nano-9b-v2",
                 vector_db: Optional[Any] = None):
        """
        Initialize the Document Analyzer Agent

        Args:
            api_key: Your NVIDIA API key from build.nvidia.com
            model: Model to use (default: nvidia/nvidia-nemotron-nano-9b-v2)
            vector_db: Optional LocalVectorDB instance for RAG capabilities
        """
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model = model
        self.max_input_length = 3000  # Words limit for nano model
        self.vector_db = vector_db

    def _call_nemotron(self, prompt: str, temperature: float = 0.3, max_tokens: int = 800,
                       use_thinking: bool = False, retry_on_empty: bool = True,
                       guided_schema: Optional[dict] = None) -> str:
        """
        Nemotron call with:
        - optional thinking
        - optional (but OFF by default) schema
        - detailed debugging of the raw response
        - safe fallbacks when content is empty/refused
        """
        try:
            # Trim long prompts (word-based)
            words = prompt.split()
            if len(words) > self.max_input_length:
                prompt = ' '.join(words[:self.max_input_length]) + "\n\n[Note: Document truncated due to length]"

            messages = [
                {"role": "system", "content": PM_SYSTEM},
                {"role": "user", "content": prompt},
            ]

            extra_body = {}
            if use_thinking:
                messages.insert(1, {"role": "system", "content": "/think"})
                extra_body.update({
                    "min_thinking_tokens": 256,
                    "max_thinking_tokens": 768
                })

            # ⚠️ TEMP: turn off schema by default (models can silently return empty).
            use_schema = bool(guided_schema and False)
            if use_schema:
                extra_body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "ConstrainedOutput", "schema": guided_schema}
                }

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False,
                extra_body=extra_body if extra_body else None
            )

            # ---- DEBUG: show raw shape so we can see what's happening ----
            finish_reason = None
            if completion and completion.choices:
                try:
                    finish_reason = completion.choices[0].finish_reason
                    if not completion.choices[0].message.content:
                        print(f"🔍 DEBUG: finish_reason={finish_reason}, content is None/empty")
                except Exception as e:
                    print(f"🔍 DEBUG: Error accessing finish_reason: {e}")

            if not completion or not completion.choices:
                if retry_on_empty:
                    time.sleep(0.2)
                    # Retry once with a simpler, shorter prompt and NO schema
                    return self._call_nemotron(
                        "Summarize the key points in 4-5 sentences.\n\n" + prompt[:1000],
                        temperature=0.2, max_tokens=250, use_thinking=False,
                        retry_on_empty=False, guided_schema=None
                    )
                return "Error: Empty response from API"

            msg = completion.choices[0].message
            content = (msg.content or "").strip()

            # Check reasoning_content (Nemotron thinking mode puts output here)
            if not content:
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    # Try to extract just the final answer, not the reasoning process
                    reasoning_text = reasoning.strip()
                    # Look for patterns that indicate the start of the actual answer
                    # Common patterns: "The summary is:", "Summary:", "In summary:", etc.
                    summary_markers = [
                        "summary:", "the summary is:", "in summary:", 
                        "here's the summary:", "summary of", "to summarize"
                    ]
                    reasoning_lower = reasoning_text.lower()
                    
                    # Try to find where the actual summary starts
                    for marker in summary_markers:
                        idx = reasoning_lower.find(marker)
                        if idx != -1:
                            # Extract from after the marker
                            content = reasoning_text[idx + len(marker):].strip()
                            # Take first paragraph or sentences
                            if "\n\n" in content:
                                content = content.split("\n\n")[0]
                            break
                    
                    # If no marker found, try to extract the last paragraph (often the final answer)
                    if not content:
                        paragraphs = [p.strip() for p in reasoning_text.split("\n\n") if p.strip()]
                        if paragraphs:
                            # Take the last substantial paragraph (likely the answer)
                            content = paragraphs[-1]
                        else:
                            # Fallback: take everything but filter obvious reasoning phrases
                            lines = reasoning_text.split("\n")
                            filtered = []
                            skip_phrases = [
                                "okay, let's", "first, i need", "i should", "i need to",
                                "the user wants", "they specified", "wait,", "need to"
                            ]
                            for line in lines:
                                line_lower = line.lower().strip()
                                if not any(phrase in line_lower for phrase in skip_phrases):
                                    filtered.append(line)
                            if filtered:
                                content = "\n".join(filtered)
                            else:
                                content = reasoning_text
                    
                    print("🔍 DEBUG: Found content in reasoning_content (extracted final answer)")

            # If model refused or gave no content, retry once in "simple mode"
            if not content:
                if retry_on_empty:
                    time.sleep(0.2)
                    # Try with even simpler prompt - just ask for text, not JSON
                    return self._call_nemotron(
                        "Summarize the key points in 4-5 sentences.\n\n" + prompt[:1000],
                        temperature=0.2, max_tokens=250, use_thinking=False,
                        retry_on_empty=False, guided_schema=None
                    )
                return "Error: No content in API response"

            return content

        except Exception as e:
            print(f"⚠️  API Error: {e}")
            if retry_on_empty:
                try:
                    # Final fallback with tiny prompt - just ask for text
                    return self._call_nemotron(
                        "Summarize the key points in 4-5 sentences.\n\n" + prompt[:1000],
                        temperature=0.2, max_tokens=250, use_thinking=False,
                        retry_on_empty=False, guided_schema=None
                    )
                except Exception:
                    return f"Error calling NVIDIA API: {e}"
            return f"Error calling NVIDIA API: {e}"

    def _rerank_chunks(self, chunks: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
        """Rerank chunks using keyword priority scoring"""
        priority_terms = ["crash", "transfer", "kyc", "fees", "abandon", "retention", "compliance",
                          "security", "biometric", "hold", "chargeback", "api", "backlog"]

        def score(chunk):
            text = chunk.get("text", "").lower()
            query_lower = query.lower()
            # Count priority term matches
            term_hits = sum(text.count(k) for k in priority_terms)
            # Boost if query terms appear
            query_hits = sum(1 for word in query_lower.split() if word in text)
            return term_hits * 2 + query_hits

        reranked = sorted(chunks, key=score, reverse=True)
        return reranked[:top_k]

    def analyze_with_rag(
        self,
        query: str,
        document_type: str = "customer_feedback",
        top_k: int = 5,
        min_similarity: float = 0.0,
        tasks: Optional[AnalysisTasks] = None,
        allow_external: bool = False
    ) -> DocumentInsight:
        """
        RAG-based document analysis: retrieves relevant chunks from vector DB and grounds analysis
        """
        if not self.vector_db:
            raise ValueError("Vector database not provided. Initialize with vector_db parameter or use analyze_document() for direct analysis.")

        print(f"🔍 RAG Mode: Retrieving relevant chunks for query...")
        print(f"   Query: {query}")

        if tasks is None:
            tasks = _infer_tasks_from_prompt(query)

        print(f"📋 Analysis tasks: summary={tasks.summary}, findings={tasks.findings}, problems={tasks.problems}, ideas={tasks.ideas}, metrics={tasks.metrics}")

        try:
            retrieved_chunks = self.vector_db.query(query, top_k=min(20, top_k * 4))

            if not retrieved_chunks:
                raise ValueError("No relevant chunks found in vector database.")

            # Prefer .txt docs, then rerank by keywords
            retrieved_chunks = _prefer_txt_first(retrieved_chunks)
            if len(retrieved_chunks) > top_k:
                retrieved_chunks = self._rerank_chunks(retrieved_chunks, query, top_k)

            context_parts, sources = [], []
            for i, chunk in enumerate(retrieved_chunks):
                doc_id = chunk.get("doc_id", "unknown")
                chunk_id = chunk.get("chunk_id", i)
                text = chunk.get("text", "")

                context_parts.append(f"[Source: {doc_id}, Chunk {chunk_id}]\n{text}")
                sources.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text
                })

            # External snippets are OFF unless explicitly allowed
            if allow_external and (len(retrieved_chunks) < max(3, top_k) or sum(len(c.get("text", "")) for c in retrieved_chunks) < 400):
                print("🔬 Local context thin, fetching external research snippets...")
                ext_snips = fetch_research_snippets(query, max_snippets=3)
                if ext_snips:
                    for j, snip in enumerate(ext_snips):
                        context_parts.append(f"[External: Web, Snippet {j+1}]\n{snip}")
                    print(f"   ✅ Added {len(ext_snips)} external snippets")

            context_text = "\n\n---\n\n".join(context_parts)

            print(f"✅ Retrieved {len(retrieved_chunks)} relevant chunks from {len(set(s['doc_id'] for s in sources))} document(s)")

        except Exception as e:
            raise ValueError(f"Error retrieving from vector database: {str(e)}")

        document_text = f"""You are given multiple evidence blocks. Each block begins with a bracketed source tag.

GROUNDING RULES:
- Use only the evidence provided here for claims, stats, and metrics.
- Prefer quantitative statements that appear in the evidence verbatim.
- Do not invent numbers. If absent, state the qualitative issue only.

RELEVANT CONTEXT:

{context_text}

---
USER QUERY: {query}

Requested sections:
- SUMMARY: {"YES" if tasks.summary else "NO"}
- KEY FINDINGS: {"YES" if tasks.findings else "NO"}
- PROBLEMS: {"YES" if tasks.problems else "NO"}
- IDEAS: {"YES" if tasks.ideas else "NO"}
- METRICS: {"YES" if tasks.metrics else "NO"}

When a section is NO, do not produce it."""

        print("\n🔄 Starting RAG-grounded analysis...\n")
        insight = self.analyze_document(document_text, document_type=document_type, tasks=tasks)

        insight.rag_enabled = True
        insight.query_used = query
        insight.retrieved_sources = sources

        if insight.metrics is not None:
            insight.metrics["rag_chunks_retrieved"] = len(retrieved_chunks)
            insight.metrics["rag_sources_count"] = len(set(s['doc_id'] for s in sources))

        # Return only requested sections (strip others)
        return self._filter_insight_for_tasks(insight, tasks)

    def analyze_document(
        self,
        document_text: str,
        document_type: str = "customer_feedback",
        tasks: Optional[AnalysisTasks] = None
    ) -> DocumentInsight:
        """
        Comprehensive document analysis using multi-step LLM calls
        """
        if tasks is None:
            tasks = _infer_tasks_from_prompt(document_text)

        print(f"📋 Analysis tasks: summary={tasks.summary}, findings={tasks.findings}, problems={tasks.problems}, ideas={tasks.ideas}, metrics={tasks.metrics}\n")

        summary = ""
        key_findings: List[str] = []
        problems: List[Dict] = []
        product_ideas: List[Dict] = []
        sentiment = {}  # no default sentiment unless tied to a flag
        metrics: Dict = {}

        # 1) SUMMARY (only if asked)
        if tasks.summary:
            print("🔄 Step 1: Generating summary (4-5 sentences)...")
            summary_prompt = f"""Write a 4-5 sentence summary of the key insights from this {document_type}. 

IMPORTANT: Return ONLY the summary text. Do not include any reasoning, thinking process, or explanations. Just the summary itself.

Evidence:
{document_text[:2500]}

Summary (4-5 sentences only):"""
            summary = (self._call_nemotron(summary_prompt, temperature=0.2, max_tokens=400) or "").strip()
            
            # Clean up summary - remove any remaining reasoning artifacts
            if summary:
                # Remove common reasoning prefixes
                reasoning_prefixes = [
                    "okay, let's", "first,", "the user wants", "they specified",
                    "i need to", "the summary needs to", "wait,", "need to present"
                ]
                summary_lower = summary.lower()
                for prefix in reasoning_prefixes:
                    if summary_lower.startswith(prefix):
                        # Find the first sentence that doesn't start with reasoning
                        sentences = summary.split('.')
                        for i, sent in enumerate(sentences):
                            sent_lower = sent.strip().lower()
                            if not any(p in sent_lower for p in reasoning_prefixes):
                                summary = '. '.join(sentences[i:]).strip()
                                break
                        break
                
                # Ensure it's actually a summary (not just reasoning)
                if any(phrase in summary.lower()[:100] for phrase in ["let's tackle", "first, i need", "the user wants"]):
                    # If it still looks like reasoning, try to extract just the summary part
                    # Look for sentences that don't contain reasoning keywords
                    sentences = [s.strip() for s in summary.split('.') if s.strip()]
                    clean_sentences = []
                    skip_words = ["okay", "let's", "first", "user wants", "they specified", "need to", "wait"]
                    for sent in sentences:
                        sent_lower = sent.lower()
                        if not any(word in sent_lower for word in skip_words):
                            clean_sentences.append(sent)
                    if clean_sentences:
                        summary = '. '.join(clean_sentences[:5]) + '.'
                    else:
                        # Last resort: take last 4-5 sentences
                        summary = '. '.join(sentences[-5:]) + '.'

        # 2) KEY FINDINGS
        if tasks.findings:
            print("🔄 Findings: extracting key findings...")
            findings_prompt = (
                "Based on the evidence below, write 2-3 sentences with concrete findings (issues, metrics, pain points). "
                "Use only information from the evidence; do not speculate.\n\n"
                f"Evidence:\n{document_text[:2500]}\n\n"
                "Key findings:"
            )

            findings_response = self._nv_chat([
                {"role": "system", "content": PM_SYSTEM},
                {"role": "user", "content": findings_prompt}
            ], max_tokens=300)

            if findings_response and not findings_response.startswith("Error"):
                sentences = [s.strip() for s in findings_response.replace('\n', ' ').split('.') if s.strip()]
                key_findings = sentences[:3]
            else:
                key_findings = []

        # 3) PROBLEMS
        if tasks.problems:
            print("🔄 Problems: identifying problems...")
            problems_prompt = FEW_SHOT_PROBLEMS + f"""Identify problems and pain points from this document.
Return ONLY a valid JSON array. Each item: "problem","severity","impact_area"
Severity: High/Medium/Low. Impact: UX/Performance/Features/Support/Compliance/Enterprise.

Document:
{document_text[:2500]}

JSON:"""
            problems_response = self._call_nemotron(problems_prompt, temperature=0.1, max_tokens=600) or "[]"
            problems = self._parse_json_objects(problems_response)

        # 4) IDEAS
        if tasks.ideas:
            print("🔄 Ideas: generating product ideas...")
            problems_summary = "\n".join([f"- {p.get('problem','')}" for p in problems[:3]]) if problems else ""
            base_prompt = FEW_SHOT_IDEAS
            if problems_summary:
                ideas_prompt = base_prompt + f"""Based on the problems below, suggest 3 product ideas.
Return ONLY a valid JSON array. Each: "title","description","impact".

Problems:
{problems_summary}

JSON:"""
            else:
                ideas_prompt = base_prompt + f"""Based on this document, suggest 3 product ideas.
Return ONLY a valid JSON array. Each: "title","description","impact".

Document:
{document_text[:2500]}

JSON:"""
            ideas_response = self._call_nemotron(ideas_prompt, temperature=0.4, max_tokens=600) or "[]"
            product_ideas = self._parse_json_objects(ideas_response)

        # 5) METRICS (only if asked)
        if tasks.metrics:
            print("🔄 Metrics: calculating...")
            metrics = self._extract_important_metrics(document_text=document_text, prompt=document_type)

        insight = DocumentInsight(
            summary=summary,
            key_findings=key_findings,
            problems_identified=problems,
            product_ideas=product_ideas,
            sentiment_analysis=sentiment,
            metrics=metrics if tasks.metrics else {},
            timestamp=datetime.now().isoformat(),
            retrieved_sources=None,
            query_used=None,
            rag_enabled=False
        )

        print("✅ Analysis complete!\n")
        return insight

    def _parse_json_list(self, response: str) -> List[str]:
        """Parse JSON array from LLM response with robust fallback and repair"""
        if response is None:
            print("⚠️  JSON parsing warning: Response is None")
            return []

        if not isinstance(response, str):
            response = str(response)

        try:
            response = response.strip()

            if response.startswith("Error"):
                print(f"⚠️  API returned error: {response}")
                return []

            # If a top-level object contains a single array field, return it
            if response.startswith("{"):
                try:
                    obj = json.loads(response)
                    if isinstance(obj, dict):
                        # Prefer 'findings' but also accept any single array field
                        if "findings" in obj and isinstance(obj["findings"], list):
                            return [str(x) for x in obj["findings"] if x]
                        array_fields = [v for v in obj.values() if isinstance(v, list)]
                        if len(array_fields) == 1:
                            return [str(x) for x in array_fields[0] if x]
                except Exception:
                    pass

            # Remove markdown code fences
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            if "[" in response:
                response = response[response.index("["):]
            if "]" in response:
                response = response[:response.rindex("]") + 1]

            parsed = json.loads(response)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
            return []

        except Exception as e:
            print(f"⚠️  JSON parsing warning: {e}, attempting repair...")
            repaired = self._repair_json_array_of_strings(response)
            return repaired if repaired else []

    def _parse_json_objects(self, response: str) -> List[Dict]:
        """Parse JSON array of objects from LLM response with robust fallback and repair"""
        if response is None:
            print("⚠️  JSON parsing warning: Response is None")
            return []

        if not isinstance(response, str):
            response = str(response)

        try:
            response = response.strip()

            if response.startswith("Error"):
                print(f"⚠️  API returned error: {response}")
                return []

            # Remove markdown
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            # Extract JSON array
            if "[" in response:
                response = response[response.index("["):]
            if "]" in response:
                response = response[:response.rindex("]") + 1]

            parsed = json.loads(response)

            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            else:
                return []

        except Exception as e:
            print(f"⚠️  JSON parsing warning: {e}, attempting repair...")
            # Determine required keys based on context (problems vs ideas)
            required_keys = ["problem", "severity", "impact_area"] if "problem" in (response or "").lower() else ["title", "description", "impact"]
            repaired = self._repair_json_array_of_objs(response, required_keys)
            return repaired

    def _parse_json_object(self, response: str) -> Dict:
        """Parse single JSON object from LLM response with fallback"""
        # Handle None or empty responses
        if response is None:
            print("⚠️  Sentiment parsing warning: Response is None")
            return {"score": 0.0, "label": "Neutral", "confidence": 0.5}

        if not isinstance(response, str):
            response = str(response)

        try:
            response = response.strip()

            # Check if response is an error message
            if response.startswith("Error"):
                print(f"⚠️  API returned error: {response}")
                return {"score": 0.0, "label": "Neutral", "confidence": 0.5}

            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            # Extract JSON object
            if "{" in response:
                response = response[response.index("{"):]
            if "}" in response:
                response = response[:response.rindex("}") + 1]

            parsed = json.loads(response)

            # Validate required keys for sentiment
            if "score" in parsed and "label" in parsed:
                return parsed
            else:
                return {"score": 0.0, "label": "Neutral", "confidence": 0.5}

        except Exception as e:
            print(f"⚠️  Sentiment parsing warning: {e}")
            return {"score": 0.0, "label": "Neutral", "confidence": 0.5}

    def _extract_metrics(self, document_text: str, problems: List[Dict], ideas: List[Dict]) -> Dict:
        """Calculate metrics from the analysis with context-aware extraction (strictly from text)"""
        # Extract dynamic metrics from context only (no defaults)
        base = _extract_context_metrics(document_text)

        # Always-safe structural counts (not "metrics" from text)
        base["total_problems"] = len(problems)
        base["high_severity_problems"] = sum(1 for p in problems if p.get("severity") == "High")
        base["medium_severity_problems"] = sum(1 for p in problems if p.get("severity") == "Medium")
        base["low_severity_problems"] = sum(1 for p in problems if p.get("severity") == "Low")
        base["total_ideas"] = len(ideas)
        base["document_length_words"] = len(document_text.split())

        return base

    def _extract_important_metrics(self, document_text: str, prompt: str) -> Dict:
        """
        Extract metrics strictly from the evidence text, then keep only those
        that matter for the user's prompt (based on keyword overlap).
        """
        # 1) Extract raw metrics already in your code
        base = _extract_context_metrics(document_text)  # returns dict of canonical metrics found

        # 2) Build relevance mask from prompt keywords
        p = (prompt or "").lower()
        interest = set()

        # map some prompt words to canonical metric keys
        interest_map = {
            "abandon": "transfer_abandon_rate",
            "retention": "new_user_retention_30d",
            "crash": "app_crash_rate",
            "biometric": "biometric_failure_rate",
            "nps": "nps",
            "api": "enterprise_api_backlog",
            "backlog": "enterprise_api_backlog",
            "hold": "compliance_hold_incidents",
            "chargeback": "chargeback_rate",
            "kyc": "kyc_failure_rate",
            "conversion": "transfer_abandon_rate",
            "support tickets": "enterprise_api_backlog",
            "tickets": "enterprise_api_backlog",
        }

        for k, canon in interest_map.items():
            if k in p:
                interest.add(canon)

        # If no explicit interest terms, keep a small, meaningful set (don't spam)
        keep = {}
        for k, v in base.items():
            if not isinstance(v, (int, float)):  # skip structural counts here
                continue
            if not interest or k in interest:
                keep[k] = v

        return keep

    def _repair_json_array_of_strings(self, s: str) -> List[str]:
        """Repair malformed JSON array of strings"""
        try:
            # Try to find and extract array
            if "[" in s and "]" in s:
                start = s.index("[")
                end = s.rindex("]") + 1
                extracted = s[start:end]
                parsed = json.loads(extracted)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item]
        except Exception:
            pass

        # Fallback: extract lines that look like findings
        lines = s.split("\n")
        findings = []
        for line in lines:
            line = line.strip().strip("-").strip("*").strip('"').strip("'").strip(",")
            if line and 10 < len(line) < 200 and not line.startswith("["):
                findings.append(line)
        return findings[:5] if findings else []

    def _repair_json_array_of_objs(self, s: str, required_keys: List[str]) -> List[Dict]:
        """Repair malformed JSON array of objects"""
        try:
            if "[" in s and "]" in s:
                start = s.index("[")
                end = s.rindex("]") + 1
                extracted = s[start:end]
                parsed = json.loads(extracted)
                if isinstance(parsed, list):
                    # Filter dicts that have required keys
                    valid = []
                    for item in parsed:
                        if isinstance(item, dict):
                            if all(k in item for k in required_keys):
                                valid.append(item)
                    return valid
        except Exception:
            pass
        return []

    def visualize_insights(self, insight: DocumentInsight, save_path: Optional[str] = None):
        """
        Generate visualizations of the analysis

        Args:
            insight: DocumentInsight object to visualize
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Product Manager Document Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Problem Severity Distribution
        severity_data = {
            'High': insight.metrics.get('high_severity_problems', 0),
            'Medium': insight.metrics.get('medium_severity_problems', 0),
            'Low': insight.metrics.get('low_severity_problems', 0)
        }

        colors = ['#ff4444', '#ffaa44', '#44ff44']
        if sum(severity_data.values()) > 0:
            axes[0, 0].bar(severity_data.keys(), severity_data.values(), color=colors)
            axes[0, 0].set_title('Problems by Severity', fontweight='bold')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].grid(axis='y', alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No problems identified',
                            ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Problems by Severity', fontweight='bold')

        # 2. Sentiment Score
        sentiment_score = insight.sentiment_analysis.get('score', 0)
        sentiment_label = insight.sentiment_analysis.get('label', 'Neutral')
        color = '#44ff44' if sentiment_score > 0.3 else '#ffaa44' if sentiment_score > -0.3 else '#ff4444'

        axes[0, 1].barh(['Sentiment'], [sentiment_score], color=color, height=0.5)
        axes[0, 1].set_xlim(-1, 1)
        axes[0, 1].set_title(f'Sentiment: {sentiment_label}', fontweight='bold')
        axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].grid(axis='x', alpha=0.3)

        # 3. Key Metrics (guard with .get)
        metrics_data = [
            insight.metrics.get('total_problems', 0),
            insight.metrics.get('high_severity_problems', 0),
            insight.metrics.get('total_ideas', 0)
        ]
        metrics_labels = ['Total\nProblems', 'High Priority\nProblems', 'Product\nIdeas']

        axes[1, 0].bar(metrics_labels, metrics_data, color=['#4488ff', '#ff4444', '#44ff88'])
        axes[1, 0].set_title('Analysis Metrics', fontweight='bold')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Text Summary Display
        axes[1, 1].axis('off')
        summary_text = f"EXECUTIVE SUMMARY\n\n{(insight.summary or '')[:280]}..."
        if insight.rag_enabled:
            summary_text += f"\n\n[RAG Mode: {insight.metrics.get('rag_chunks_retrieved', 0)} chunks from {insight.metrics.get('rag_sources_count', 0)} source(s)]"
        axes[1, 1].text(0.05, 0.95, summary_text, wrap=True, fontsize=9,
                        verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        axes[1, 1].set_title('Summary Preview', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")

        plt.show()

    def visualize_metrics(self, insight: DocumentInsight, save_path: Optional[str] = None):
        """
        Generate a focused metrics visualization (bar chart of KPIs)
        
        Args:
            insight: DocumentInsight object with metrics
            save_path: Path to save the PNG file
        """
        import matplotlib.pyplot as plt
        
        # Filter out RAG metadata and structural counts, keep only actual metrics
        metrics_to_plot = {}
        skip_keys = ['rag_chunks_retrieved', 'rag_sources_count', 'total_problems', 
                     'high_severity_problems', 'medium_severity_problems', 
                     'low_severity_problems', 'total_ideas', 'document_length_words']
        
        for k, v in (insight.metrics or {}).items():
            if k not in skip_keys and isinstance(v, (int, float)):
                # Format metric names for display
                display_name = k.replace('_', ' ').title()
                metrics_to_plot[display_name] = v
        
        if not metrics_to_plot:
            print("⚠️  No metrics available to visualize")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        metric_names = list(metrics_to_plot.keys())
        metric_values = list(metrics_to_plot.values())
        
        # Convert percentages to readable format (if values are 0-1, show as %)
        display_values = []
        for i, val in enumerate(metric_values):
            if 0 <= val <= 1 and 'rate' in metric_names[i].lower():
                display_values.append(val * 100)  # Show as percentage
            else:
                display_values.append(val)
        
        # Create bar chart
        colors = plt.cm.viridis([i / len(metric_names) for i in range(len(metric_names))])
        bars = ax.bar(metric_names, display_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # Customize chart
        ax.set_title('Key Performance Indicators (KPIs)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, val, orig_val) in enumerate(zip(bars, display_values, metric_values)):
            height = bar.get_height()
            metric_name = metric_names[i]  # Get the metric name for this bar
            
            # Format label based on value type
            if 0 <= orig_val <= 1 and 'rate' in metric_name.lower():
                label = f'{val:.1f}%'
            elif isinstance(orig_val, float):
                label = f'{val:.3f}'
            else:
                label = f'{int(val)}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Metrics visualization saved to {save_path}")
        
        # Don't show interactively (we're just saving)
        plt.close(fig)

    def export_to_json(self, insight: DocumentInsight, filepath: str):
        """Export analysis results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(insight), f, indent=2, ensure_ascii=False)
        print(f"💾 Analysis exported to {filepath}")

    def generate_report(self, insight: DocumentInsight) -> str:
        """Generate a formatted text report"""
        parts = []

        header = f"""
{'='*80}
PRODUCT MANAGER DOCUMENT ANALYSIS REPORT
Generated: {insight.timestamp}
Model: NVIDIA Nemotron Nano 9B v2
{'='*80}
"""
        parts.append(header)

        # RAG header if enabled
        if insight.rag_enabled:
            rag_header = f"""
🔍 RAG ANALYSIS MODE
{'-'*80}
Query: {insight.query_used}
Retrieved Chunks: {insight.metrics.get('rag_chunks_retrieved', 0) if insight.metrics else 0}
Sources: {insight.metrics.get('rag_sources_count', 0) if insight.metrics else 0} document(s)

📚 SOURCES USED:
{'-'*80}
"""
            for i, source in enumerate(insight.retrieved_sources or [], 1):
                rag_header += f"{i}. Document: {source.get('doc_id', 'unknown')}, Chunk {source.get('chunk_id', 'N/A')}\n"
                rag_header += f"   Preview: {source.get('text_preview', 'N/A')}\n\n"
            parts.append(rag_header)

        # Summary (only if present)
        if insight.summary:
            parts.append(f"""
📋 EXECUTIVE SUMMARY
{'-'*80}
{insight.summary}
""")

        # Key Findings (only if present)
        if insight.key_findings:
            body = "\n".join(f"{i+1}. {f}" for i, f in enumerate(insight.key_findings))
            parts.append(f"""
🔍 KEY FINDINGS ({len(insight.key_findings)} items)
{'-'*80}
{body}
""")

        # Problems (only if present)
        if insight.problems_identified:
            total = len(insight.problems_identified)
            high = sum(1 for p in insight.problems_identified if p.get('severity') == 'High')
            parts.append(f"""
⚠️  PROBLEMS IDENTIFIED ({total} total, {high} high priority)
{'-'*80}
""")
            for i, p in enumerate(insight.problems_identified, 1):
                sev = p.get('severity', 'N/A')
                emoji = "🔴" if sev == "High" else "🟡" if sev == "Medium" else "🟢"
                parts.append(f"{i}. {emoji} [{sev}] {p.get('problem', 'N/A')}\n   📂 Impact Area: {p.get('impact_area', 'N/A')}\n")

        # Product Ideas (only if present)
        if insight.product_ideas:
            parts.append(f"""
💡 PRODUCT IDEAS & SOLUTIONS ({len(insight.product_ideas)} ideas)
{'-'*80}
""")
            for i, idea in enumerate(insight.product_ideas, 1):
                parts.append(f"{i}. �� {idea.get('title', 'Untitled')}\n   📝 {idea.get('description', '')}\n   📈 Impact: {idea.get('impact', '')}\n")

        # Sentiment (only if analyzed)
        if insight.sentiment_analysis and insight.sentiment_analysis.get('label') != 'Neutral':
            sentiment = insight.sentiment_analysis
            sentiment_emoji = "😊" if sentiment.get('score', 0) > 0.3 else "😐" if sentiment.get('score', 0) > -0.3 else "😟"
            parts.append(f"""
💭 SENTIMENT ANALYSIS
{'-'*80}
{sentiment_emoji} Overall Sentiment: {sentiment.get('label', 'N/A')} (Score: {sentiment.get('score', 0):.2f})
📊 Confidence: {sentiment.get('confidence', 0):.0%}
""")

        # Metrics (only if present)
        if insight.metrics:
            parts.append(f"""
📊 DOCUMENT METRICS
{'-'*80}
""")
            for k, v in insight.metrics.items():
                if k not in ['rag_chunks_retrieved', 'rag_sources_count']:  # RAG metrics shown in header
                    parts.append(f"• {k.replace('_', ' ').title()}: {v}")
            parts.append("")

        parts.append(f"{'='*80}\n")
        return "\n".join(parts)

    def _nv_chat(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """
        Simple chat wrapper that handles Nemotron's quirks
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
                top_p=0.95,
                stream=False,
            )

            if not completion or not completion.choices:
                return ""

            msg = completion.choices[0].message
            content = (msg.content or "").strip()

            # Check reasoning_content
            if not content:
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    content = reasoning.strip()

            return content
        except Exception as e:
            print(f"⚠️  Chat error: {e}")
            return ""

    def _filter_insight_for_tasks(self, insight: DocumentInsight, tasks: AnalysisTasks) -> DocumentInsight:
        """Strip unrequested sections before returning/exporting"""
        from copy import deepcopy
        out = deepcopy(insight)

        if not tasks.summary:
            out.summary = ""
        if not tasks.findings:
            out.key_findings = []
        if not tasks.problems:
            out.problems_identified = []
        if not tasks.ideas:
            out.product_ideas = []
        if not tasks.metrics:
            out.metrics = {}

        # sentiment left empty; you can tie it behind a flag later if you wish
        if not (tasks.findings or tasks.problems):
            out.sentiment_analysis = {}

        return out


# Example Usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    # Load .env from ai-agents directory (parent of agents/)
    ai_agents_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(ai_agents_dir, ".env"))

    # Get API key from environment variable instead of hardcoding
    API_KEY = os.getenv("NVIDIA_API_KEY")

    if not API_KEY:
        raise ValueError("NVIDIA_API_KEY environment variable is not set. Please set it in your .env file.")

    # Initialize the agent
    print("🤖 Initializing Nemotron Document Analyzer Agent...")
    agent = NemotronDocumentAnalyzer(api_key=API_KEY)

    # Sample customer feedback document
    sample_document = """
    Customer Feedback Summary - Q4 2024

    Our mobile app has received significant feedback from users. Many customers report
    that the checkout process takes too long, with an average of 7 steps to complete
    a purchase. Users also mention frequent crashes when uploading images, particularly
    on Android devices running version 12 and above.

    On the positive side, customers love the new recommendation engine - it has increased
    engagement by 40%. The dark mode feature has been highly praised, with 85% of users
    enabling it within their first session.

    However, there's growing frustration with the search functionality. Users report that
    search results are often irrelevant, and the lack of filters makes it difficult to
    find specific products. Customer support tickets related to search have increased
    by 120% this quarter.

    Several enterprise clients have requested bulk upload capabilities and better API
    documentation. They're willing to pay premium prices for these features.

    The user onboarding experience needs improvement. New users struggle to understand
    key features, and 30% abandon the app within the first week. Better tutorials and
    guided tours could significantly improve retention rates.
    """

    print("\n" + "=" * 80)
    print("🔬 Starting Document Analysis...")
    print("=" * 80 + "\n")

    # Analyze the document
    insights = agent.analyze_document(sample_document, document_type="customer_feedback")

    # Generate and print report
    report = agent.generate_report(insights)
    print(report)

    # Export results
    agent.export_to_json(insights, "analysis_results.json")

    # Create visualizations
    try:
        agent.visualize_insights(insights, save_path="analysis_dashboard.png")
    except Exception as e:
        print(f"⚠️  Visualization skipped: {e}")

    print("\n✅ Analysis complete!")
    print("📁 Files created:")
    print("   • analysis_results.json - Full analysis data")
    print("   • analysis_dashboard.png - Visual dashboard")
