import os
from dotenv import load_dotenv
from agents.customerAgent import NemotronDocumentAnalyzer, _infer_tasks_from_prompt, AnalysisTasks
from storage.vector_db import LocalVectorDB
from agents.research import fetch_research_snippets

# Load .env from ai-agents directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

API_KEY = os.getenv("NVIDIA_API_KEYX")
LLM_MODEL = os.getenv("NVIDIA_LLM_MODEX", "nvidia/nvidia-nemotron-nano-9b-v2")
EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODELX")
DB_PATH = os.getenv("VECTOR_DB_PATH", "storage/local_vectors.json")

# Helper function to detect if query asks for visualization
def _wants_visualization(query: str) -> bool:
    """Check if query contains keywords requesting charts/graphs/visualizations"""
    viz_keywords = ("chart", "graph", "visualize", "visualization", "png", "image", 
                   "plot", "dashboard", "kpi", "kpis", "visual", "diagram")
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in viz_keywords)

# Initialize vector database and agent with RAG support
print("🤖 Initializing RAG Pipeline...")
retriever = LocalVectorDB(api_key=API_KEY, embed_model=EMBED_MODEL, db_path=DB_PATH)
agent = NemotronDocumentAnalyzer(api_key=API_KEY, model=LLM_MODEL, vector_db=retriever)

query = os.getenv("USER_QUERY", "Give key findings and 3 product ideas; skip metrics and summary")

print(f"\n🔍 Query: {query}\n")

tasks = _infer_tasks_from_prompt(query)

insight = agent.analyze_with_rag(
    query=query,
    document_type="customer_feedback",
    top_k=5,
    tasks=tasks,
    allow_external=False,  # keep RAG local to embedded .txt by default
)

os.makedirs("data/results", exist_ok=True)
agent.export_to_json(insight, "data/results/rag_analysis.json")

# Check if visualization is requested
wants_viz = _wants_visualization(query)

# Print only what the user asked for
if tasks.summary and not (tasks.findings or tasks.problems or tasks.ideas or tasks.metrics):
    print("\n📋 Summary:\n", insight.summary or "(none)")
elif tasks.findings and not (tasks.summary or tasks.problems or tasks.ideas or tasks.metrics):
    print("🔎 Key Findings:")
    for i, f in enumerate(insight.key_findings, 1):
        print(f"   {i}. {f}")
elif tasks.problems and not (tasks.summary or tasks.findings or tasks.ideas or tasks.metrics):
    print("⚠️ Problems:")
    for i, p in enumerate(insight.problems_identified, 1):
        print(f"   {i}. [{p.get('severity','?')}/{p.get('impact_area','?')}] {p.get('problem','')}")
elif tasks.ideas and not (tasks.summary or tasks.findings or tasks.problems or tasks.metrics):
    print("💡 Ideas:")
    for i, idea in enumerate(insight.product_ideas, 1):
        print(f"   {i}. {idea.get('title','')} — {idea.get('impact','')}")
elif tasks.metrics and not (tasks.summary or tasks.findings or tasks.problems or tasks.ideas):
    print("📊 Metrics:")
    for k, v in (insight.metrics or {}).items():
        print(f"   • {k}: {v}")
    
    # Auto-generate visualization when metrics are requested
    if insight.metrics:
        try:
            viz_path = "data/results/metrics_chart.png"
            agent.visualize_metrics(insight, save_path=viz_path)
            print(f"\n📊 Metrics chart saved to {viz_path}")
        except Exception as e:
            print(f"⚠️  Could not generate metrics chart: {e}")
else:
    # Mixed request: output a compact bundle
    if tasks.summary and insight.summary:
        print("\n📋 Summary:\n", insight.summary)
    if tasks.findings and insight.key_findings:
        print("\n🔎 Key Findings:")
        for i, f in enumerate(insight.key_findings, 1):
            print(f"   {i}. {f}")
    if tasks.problems and insight.problems_identified:
        print("\n⚠️ Problems:")
        for i, p in enumerate(insight.problems_identified, 1):
            print(f"   {i}. [{p.get('severity','?')}/{p.get('impact_area','?')}] {p.get('problem','')}")
    if tasks.ideas and insight.product_ideas:
        print("\n💡 Ideas:")
        for i, idea in enumerate(insight.product_ideas, 1):
            print(f"   {i}. {idea.get('title','')} — {idea.get('impact','')}")
    if tasks.metrics and insight.metrics:
        print("\n📊 Metrics:")
        for k, v in insight.metrics.items():
            print(f"   • {k}: {v}")
        
        # Auto-generate visualization when metrics are requested
        try:
            viz_path = "data/results/metrics_chart.png"
            agent.visualize_metrics(insight, save_path=viz_path)
            print(f"\n📊 Metrics chart saved to {viz_path}")
        except Exception as e:
            print(f"⚠️  Could not generate metrics chart: {e}")

print("\n✅ RAG Analysis complete!")
