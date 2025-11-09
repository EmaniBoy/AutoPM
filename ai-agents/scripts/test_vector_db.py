import os
from dotenv import load_dotenv
from storage.vector_db import LocalVectorDB

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEYX")
EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODELX")
DB_PATH = os.getenv("VECTOR_DB_PATH")

if not API_KEY:
    raise ValueError("NVIDIA_API_KEYX not found in environment variables. Please set it in your .env file.")

if not EMBED_MODEL:
    raise ValueError("NVIDIA_EMBED_MODELX not found in environment variables. Please set it in your .env file.")

db = LocalVectorDB(api_key=API_KEY, embed_model=EMBED_MODEL, db_path=DB_PATH)

# Add two sample chunks
chunks = [
    "Our users complain about long checkout times.",
    "The app crashes frequently when uploading images."
]
db.add_document("sample_feedback", chunks)

# Query for something semantically similar
query = "checkout experience issues"
results = db.query(query, top_k=2)
for r in results:
    print("\n🔎 Match:")
    print(f"Doc: {r['doc_id']} | Chunk: {r['chunk_id']}")
    print(f"Text: {r['text']}")
