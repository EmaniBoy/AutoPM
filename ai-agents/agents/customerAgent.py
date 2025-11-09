import os
import json
from typing import Dict, List, Optional, Union
from openai import OpenAI
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from datetime import datetime

@dataclass
class DocumentInsight:
    """Structure for document analysis insights"""
    summary: str
    key_findings: List[str]
    problems_identified: List[Dict[str, str]]
    product_ideas: List[Dict[str, str]]
    sentiment_analysis: Dict[str, any]
    metrics: Dict[str, any]
    timestamp: str
    # RAG-specific fields
    retrieved_sources: Optional[List[Dict[str, str]]] = None
    query_used: Optional[str] = None
    rag_enabled: bool = False

class NemotronDocumentAnalyzer:
    """
    AI Agent for analyzing customer data and documents using NVIDIA Nemotron LLM
    Optimized for nvidia-nemotron-nano-9b-v2
    Supports both direct document analysis and RAG-based analysis
    """
    
    def __init__(self, api_key: str, model: str = "nvidia/nvidia-nemotron-nano-9b-v2", 
                 vector_db: Optional[any] = None):
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
    
    def _call_nemotron(self, prompt: str, temperature: float = 0.6, max_tokens: int = 1500, 
                       use_thinking: bool = False) -> str:
        """
        Make API call to NVIDIA Nemotron LLM
        
        Args:
            prompt: The prompt to send to the model
            temperature: Controls randomness (0.0-1.0)
            max_tokens: Maximum response length
            use_thinking: Enable reasoning tokens for complex tasks
            
        Returns:
            Model response text
        """
        try:
            # Truncate prompt if too long
            words = prompt.split()
            if len(words) > self.max_input_length:
                prompt = ' '.join(words[:self.max_input_length])
                prompt += "\n\n[Note: Document truncated due to length]"
            
            messages = [{"role": "user", "content": prompt}]
            
            # Add thinking capability for complex reasoning
            extra_body = {}
            if use_thinking:
                messages.insert(0, {"role": "system", "content": "/think"})
                extra_body = {
                    "min_thinking_tokens": 512,
                    "max_thinking_tokens": 1024
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
                extra_body=extra_body
            )
            
            # Handle None response
            if not completion or not completion.choices or len(completion.choices) == 0:
                return "Error: Empty response from API"
            
            content = completion.choices[0].message.content
            if content is None:
                return "Error: No content in API response"
            
            return content
            
        except Exception as e:
            print(f"⚠️  API Error: {str(e)}")
            return f"Error calling NVIDIA API: {str(e)}"
    
    def analyze_with_rag(self, query: str, document_type: str = "customer_feedback", 
                        top_k: int = 5, min_similarity: float = 0.0) -> DocumentInsight:
        """
        RAG-based document analysis: retrieves relevant chunks from vector DB and grounds analysis
        
        Args:
            query: User query/question to analyze
            document_type: Type of document (customer_feedback, survey, support_tickets, etc.)
            top_k: Number of relevant chunks to retrieve
            min_similarity: Minimum similarity threshold for retrieved chunks
            
        Returns:
            DocumentInsight object with RAG-grounded analysis
        """
        if not self.vector_db:
            raise ValueError("Vector database not provided. Initialize with vector_db parameter or use analyze_document() for direct analysis.")
        
        print(f"🔍 RAG Mode: Retrieving relevant chunks for query...")
        print(f"   Query: {query}")
        
        # Step 1: Retrieve relevant chunks from vector database
        try:
            retrieved_chunks = self.vector_db.query(query, top_k=top_k)
            
            if not retrieved_chunks:
                raise ValueError("No relevant chunks found in vector database. Please ensure documents are indexed.")
            
            # Format retrieved context with source citations
            context_parts = []
            sources = []
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
            
            context_text = "\n\n---\n\n".join(context_parts)
            
            print(f"✅ Retrieved {len(retrieved_chunks)} relevant chunks from {len(set(s['doc_id'] for s in sources))} document(s)")
            
        except Exception as e:
            raise ValueError(f"Error retrieving from vector database: {str(e)}")
        
        # Step 2: Create RAG-grounded document text
        document_text = f"""RELEVANT CONTEXT FROM DOCUMENT DATABASE:

{context_text}

---
USER QUERY: {query}

Based on the retrieved context above, analyze and provide insights. When referencing specific information, cite the source document and chunk number."""
        
        # Step 3: Perform analysis with RAG context
        print("\n🔄 Starting RAG-grounded analysis...\n")
        insight = self.analyze_document(document_text, document_type=document_type)
        
        # Step 4: Add RAG metadata
        insight.rag_enabled = True
        insight.query_used = query
        insight.retrieved_sources = sources
        
        # Update metrics to include RAG info
        insight.metrics["rag_chunks_retrieved"] = len(retrieved_chunks)
        insight.metrics["rag_sources_count"] = len(set(s['doc_id'] for s in sources))
        
        return insight
    
    def analyze_document(self, document_text: str, document_type: str = "customer_feedback") -> DocumentInsight:
        """
        Comprehensive document analysis using multi-step LLM calls
        Can work with direct document text or RAG-retrieved context
        
        Args:
            document_text: The text content to analyze
            document_type: Type of document (customer_feedback, survey, support_tickets, etc.)
            
        Returns:
            DocumentInsight object with structured analysis
        """
        
        print("🔄 Step 1/6: Generating executive summary...")
        # Step 1: Generate Executive Summary
        summary_prompt = f"""You are a product manager analyzing {document_type}. 
Write a 3-sentence executive summary highlighting the most critical insights.

Document:
{document_text[:2500]}

Provide only the summary, no preamble:"""
        
        summary = self._call_nemotron(summary_prompt, temperature=0.3, max_tokens=300)
        
        print("🔄 Step 2/6: Extracting key findings...")
        # Step 2: Extract Key Findings
        findings_prompt = f"""Extract 5 key findings from this document.
Return ONLY a valid JSON array of strings. No explanation, just the array.

Document:
{document_text[:2500]}

Format: ["finding 1", "finding 2", "finding 3", "finding 4", "finding 5"]
JSON:"""
        
        findings_response = self._call_nemotron(findings_prompt, temperature=0.2, max_tokens=500)
        key_findings = self._parse_json_list(findings_response)
        
        print("🔄 Step 3/6: Identifying problems...")
        # Step 3: Identify Problems with thinking enabled for better reasoning
        problems_prompt = f"""Identify problems and pain points from this document.
Return ONLY a valid JSON array. Each problem must have: "problem", "severity", "impact_area"

Severity options: "High", "Medium", "Low"
Impact area options: "UX", "Performance", "Features", "Support", "Security", "Other"

Document:
{document_text[:2500]}

Format: [{{"problem": "description", "severity": "High", "impact_area": "UX"}}]
JSON:"""
        
        problems_response = self._call_nemotron(problems_prompt, temperature=0.2, max_tokens=800, use_thinking=True)
        problems = self._parse_json_objects(problems_response)
        
        # Ensure we have at least some problems
        if not problems:
            problems = [{
                "problem": "Unable to extract structured problems from document",
                "severity": "Medium",
                "impact_area": "Other"
            }]
        
        print("🔄 Step 4/6: Generating product ideas...")
        # Step 4: Generate Product Ideas
        problems_summary = "\n".join([f"- {p.get('problem', 'N/A')} [{p.get('severity', 'N/A')}]" 
                                      for p in problems[:3]])
        
        ideas_prompt = f"""Based on these problems, suggest 3 innovative product ideas/solutions.
Return ONLY a valid JSON array. Each idea must have: "title", "description", "impact"

Problems identified:
{problems_summary}

Format: [{{"title": "Feature Name", "description": "Brief description", "impact": "Expected outcome"}}]
JSON:"""
        
        ideas_response = self._call_nemotron(ideas_prompt, temperature=0.6, max_tokens=800, use_thinking=True)
        product_ideas = self._parse_json_objects(ideas_response)
        
        # Fallback ideas if parsing fails
        if not product_ideas:
            product_ideas = [{
                "title": "Enhanced User Experience",
                "description": "Address identified UX issues with improved interface design",
                "impact": "Reduced user friction and improved satisfaction"
            }]
        
        print("🔄 Step 5/6: Analyzing sentiment...")
        # Step 5: Sentiment Analysis
        sentiment_prompt = f"""Analyze the sentiment of this feedback.
Return ONLY a valid JSON object with: "score", "label", "confidence"

Rules:
- score: number from -1.0 (very negative) to 1.0 (very positive)
- label: "Positive", "Neutral", or "Negative"
- confidence: number from 0.0 to 1.0

Document:
{document_text[:2000]}

Format: {{"score": 0.5, "label": "Positive", "confidence": 0.8}}
JSON:"""
        
        sentiment_response = self._call_nemotron(sentiment_prompt, temperature=0.1, max_tokens=200)
        sentiment = self._parse_json_object(sentiment_response)
        
        print("🔄 Step 6/6: Calculating metrics...")
        # Step 6: Extract Metrics
        metrics = self._extract_metrics(document_text, problems, product_ideas)
        
        # Create structured insight
        insight = DocumentInsight(
            summary=summary.strip(),
            key_findings=key_findings,
            problems_identified=problems,
            product_ideas=product_ideas,
            sentiment_analysis=sentiment,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            retrieved_sources=None,
            query_used=None,
            rag_enabled=False
        )
        
        print("✅ Analysis complete!\n")
        return insight
    
    def _parse_json_list(self, response: str) -> List[str]:
        """Parse JSON array from LLM response with robust fallback"""
        # Handle None or empty responses
        if response is None:
            print("⚠️  JSON parsing warning: Response is None")
            return ["Analysis completed but no structured data available"]
        
        if not isinstance(response, str):
            response = str(response)
        
        try:
            # Clean the response
            response = response.strip()
            
            # Check if response is an error message
            if response.startswith("Error"):
                print(f"⚠️  API returned error: {response}")
                return ["Unable to extract findings due to API error"]
            
            # Remove markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            # Remove any preamble text before the JSON
            if "[" in response:
                response = response[response.index("["):]
            if "]" in response:
                response = response[:response.rindex("]")+1]
            
            parsed = json.loads(response)
            
            # Validate it's actually a list
            if isinstance(parsed, list):
                # Ensure all items are strings
                return [str(item) for item in parsed if item]
            else:
                return []
                
        except Exception as e:
            print(f"⚠️  JSON parsing warning: {e}")
            # Fallback: extract lines that look like findings
            if not response:
                return ["Analysis completed but structured data unavailable"]
            
            lines = response.split("\n")
            findings = []
            for line in lines:
                line = line.strip().strip("-").strip("*").strip('"').strip("'")
                if line and len(line) > 10 and len(line) < 200:
                    findings.append(line)
            return findings[:5] if findings else ["Analysis completed but structured data unavailable"]
    
    def _parse_json_objects(self, response: str) -> List[Dict]:
        """Parse JSON array of objects from LLM response with robust fallback"""
        # Handle None or empty responses
        if response is None:
            print("⚠️  JSON parsing warning: Response is None")
            return []
        
        if not isinstance(response, str):
            response = str(response)
        
        try:
            response = response.strip()
            
            # Check if response is an error message
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
                response = response[:response.rindex("]")+1]
            
            parsed = json.loads(response)
            
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            else:
                return []
                
        except Exception as e:
            print(f"⚠️  JSON parsing warning: {e}")
            return []
    
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
                response = response[:response.rindex("}")+1]
            
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
        """Calculate metrics from the analysis"""
        high_priority = sum(1 for p in problems if p.get("severity") == "High")
        
        return {
            "total_problems": len(problems),
            "high_severity_problems": high_priority,
            "medium_severity_problems": sum(1 for p in problems if p.get("severity") == "Medium"),
            "low_severity_problems": sum(1 for p in problems if p.get("severity") == "Low"),
            "total_ideas": len(ideas),
            "document_length_words": len(document_text.split()),
            "analysis_completeness": 1.0
        }
    
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
        
        # 3. Key Metrics
        metrics_data = [
            insight.metrics['total_problems'],
            insight.metrics['high_severity_problems'],
            insight.metrics['total_ideas']
        ]
        metrics_labels = ['Total\nProblems', 'High Priority\nProblems', 'Product\nIdeas']
        
        axes[1, 0].bar(metrics_labels, metrics_data, color=['#4488ff', '#ff4444', '#44ff88'])
        axes[1, 0].set_title('Analysis Metrics', fontweight='bold')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Text Summary Display
        axes[1, 1].axis('off')
        summary_text = f"EXECUTIVE SUMMARY\n\n{insight.summary[:280]}..."
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
    
    def export_to_json(self, insight: DocumentInsight, filepath: str):
        """Export analysis results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(insight), f, indent=2, ensure_ascii=False)
        print(f"💾 Analysis exported to {filepath}")
    
    def generate_report(self, insight: DocumentInsight) -> str:
        """Generate a formatted text report"""
        rag_header = ""
        if insight.rag_enabled:
            rag_header = f"""
🔍 RAG ANALYSIS MODE
{'-'*80}
Query: {insight.query_used}
Retrieved Chunks: {insight.metrics.get('rag_chunks_retrieved', 0)}
Sources: {insight.metrics.get('rag_sources_count', 0)} document(s)

📚 SOURCES USED:
{'-'*80}
"""
            for i, source in enumerate(insight.retrieved_sources or [], 1):
                rag_header += f"{i}. Document: {source.get('doc_id', 'unknown')}, Chunk {source.get('chunk_id', 'N/A')}\n"
                rag_header += f"   Preview: {source.get('text_preview', 'N/A')}\n\n"
        
        report = f"""
{'='*80}
PRODUCT MANAGER DOCUMENT ANALYSIS REPORT
Generated: {insight.timestamp}
Model: NVIDIA Nemotron Nano 9B v2
{'='*80}
{rag_header}
📋 EXECUTIVE SUMMARY
{'-'*80}
{insight.summary}

🔍 KEY FINDINGS ({len(insight.key_findings)} items)
{'-'*80}
"""
        for i, finding in enumerate(insight.key_findings, 1):
            report += f"{i}. {finding}\n"
        
        total_problems = insight.metrics['total_problems']
        high_priority = insight.metrics['high_severity_problems']
        
        report += f"""
⚠️  PROBLEMS IDENTIFIED ({total_problems} total, {high_priority} high priority)
{'-'*80}
"""
        for i, problem in enumerate(insight.problems_identified, 1):
            severity_emoji = "🔴" if problem.get('severity') == 'High' else "🟡" if problem.get('severity') == 'Medium' else "🟢"
            report += f"\n{i}. {severity_emoji} [{problem.get('severity', 'N/A')}] {problem.get('problem', 'N/A')}\n"
            report += f"   📂 Impact Area: {problem.get('impact_area', 'N/A')}\n"
        
        report += f"""
💡 PRODUCT IDEAS & SOLUTIONS ({len(insight.product_ideas)} ideas)
{'-'*80}
"""
        for i, idea in enumerate(insight.product_ideas, 1):
            report += f"\n{i}. 🚀 {idea.get('title', 'Untitled')}\n"
            report += f"   📝 {idea.get('description', 'No description')}\n"
            report += f"   📈 Impact: {idea.get('impact', 'N/A')}\n"
        
        sentiment = insight.sentiment_analysis
        sentiment_emoji = "😊" if sentiment.get('score', 0) > 0.3 else "😐" if sentiment.get('score', 0) > -0.3 else "😟"
        
        report += f"""
💭 SENTIMENT ANALYSIS
{'-'*80}
{sentiment_emoji} Overall Sentiment: {sentiment.get('label', 'N/A')} (Score: {sentiment.get('score', 0):.2f})
📊 Confidence: {sentiment.get('confidence', 0):.0%}

📊 DOCUMENT METRICS
{'-'*80}
• Document Length: {insight.metrics['document_length_words']} words
• Total Problems: {insight.metrics['total_problems']}
• High Priority: {insight.metrics['high_severity_problems']}
• Medium Priority: {insight.metrics['medium_severity_problems']}
• Low Priority: {insight.metrics['low_severity_problems']}
• Product Ideas Generated: {insight.metrics['total_ideas']}
"""
        if insight.rag_enabled:
            report += f"""• RAG Chunks Retrieved: {insight.metrics.get('rag_chunks_retrieved', 0)}
• RAG Sources: {insight.metrics.get('rag_sources_count', 0)}
"""
        
        report += f"""
{'='*80}
"""
        return report


# Example Usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
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
    
    print("\n" + "="*80)
    print("🔬 Starting Document Analysis...")
    print("="*80 + "\n")
    
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