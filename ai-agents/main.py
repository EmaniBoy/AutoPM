import os
from dotenv import load_dotenv
from agents.customerAgent import NemotronDocumentAnalyzer
from storage.vector_db import LocalVectorDB

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")
EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL")
DB_PATH = os.getenv("VECTOR_DB_PATH", "storage/local_vectors.json")

# Initialize vector database and agent with RAG support
print("🤖 Initializing RAG Pipeline...")
retriever = LocalVectorDB(api_key=API_KEY, embed_model=EMBED_MODEL, db_path=DB_PATH)
agent = NemotronDocumentAnalyzer(api_key=API_KEY, model=LLM_MODEL, vector_db=retriever)

# User query or doc topic
query = "customer feedback issues related to checkout experience"

print(f"\n🔍 Query: {query}\n")

# Use RAG-based analysis (retrieves relevant chunks automatically)
insight = agent.analyze_with_rag(
    query=query, 
    document_type="customer_feedback",
    top_k=5
)

# Generate and print report
report = agent.generate_report(insight)
print(report)

# Save & visualize - ensure directory exists
os.makedirs("data/results", exist_ok=True)
agent.export_to_json(insight, "data/results/rag_analysis.json")
agent.visualize_insights(insight, save_path="data/results/rag_dashboard.png")

print("\n✅ RAG Analysis complete!")
