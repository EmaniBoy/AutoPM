import os
import sys
from dotenv import load_dotenv

# Add parent directory to Python path so we can import storage and other modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.vector_db import LocalVectorDB

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL")
DB_PATH = os.getenv("VECTOR_DB_PATH", "storage/local_vectors.json")

if not API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in environment variables. Please set it in your .env file.")

if not EMBED_MODEL:
    raise ValueError("NVIDIA_EMBED_MODEL not found in environment variables. Please set it in your .env file.")

def chunk_text(text, size=800, overlap=150):
    """
    Split text into overlapping chunks for better context preservation
    """
    words = text.split()
    chunks = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i+size]))
    return chunks

if __name__ == "__main__":
    # Get the ai-agents directory (parent of scripts)
    ai_agents_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(ai_agents_dir, "data", "raw")
    
    if not os.path.exists(raw_dir):
        print(f"⚠️  Directory '{raw_dir}' does not exist. Creating it...")
        os.makedirs(raw_dir, exist_ok=True)
        print(f"✅ Created '{raw_dir}'. Please add .txt files to index.")
        sys.exit(0)

    # Collect .txt files
    files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
    if not files:
        print(f"⚠️  No .txt files found in '{raw_dir}'. Please add documents to index.")
        sys.exit(0)

    print("🤖 Initializing Vector Database...")
    try:
        # Use absolute path for DB_PATH if it's relative
        if not os.path.isabs(DB_PATH):
            DB_PATH = os.path.join(ai_agents_dir, DB_PATH)
        db = LocalVectorDB(api_key=API_KEY, embed_model=EMBED_MODEL, db_path=DB_PATH)
    except Exception as e:
        print(f"❌ Error initializing vector database: {e}")
        sys.exit(1)

    print(f"📂 Scanning '{raw_dir}' for documents...\n")

    indexed_count = 0
    error_count = 0

    for fname in sorted(files):
        try:
            filepath = os.path.join(raw_dir, fname)
            doc_id = fname.rsplit(".", 1)[0]

            print(f"📄 Processing: {fname}...")

            # Read file with encoding handling
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(filepath, "r", encoding="latin-1") as f:
                    text = f.read()

            if not text.strip():
                print(f"   ⚠️  Skipping empty file: {fname}")
                continue

            # Chunk the document
            chunks = chunk_text(text)
            if not chunks:
                print(f"   ⚠️  No chunks created from: {fname}")
                continue

            # Add to vector database
            db.add_document(doc_id, chunks)
            indexed_count += 1
            print(f"   ✅ Indexed {fname} ({len(chunks)} chunks)\n")

        except Exception as e:
            print(f"   ❌ Error processing {fname}: {e}\n")
            error_count += 1
            continue

    print("=" * 60)
    print("✅ Indexing complete!")
    print(f"   • Documents indexed: {indexed_count}")
    print(f"   • Errors: {error_count}")
    print(f"   • Total chunks: {len(db.vectors)}")
    print(f"   • Database saved to: {DB_PATH}")
    print("=" * 60)
