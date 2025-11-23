import os
import uuid
from repo_loader import RepoLoader
from code_extractor import CodeExtractor
from repo_intel import RepoIntel
from embeddings import EmbeddingStoreFAISS
try:
    from search import CodeSearchFAISS
except ImportError:
    CodeSearchFAISS = None
from backend.app.team_adapter import save_for_graphflow
from dotenv import load_dotenv

# Load API key from .env (better than hardcoding)
load_dotenv()

# Fallback if .env doesn't have it
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY_HERE"

def process_repository(file_path: str):

    print(f"Loading repository from: {file_path}")
    # 1️⃣ Load repo and documents
    repo_path = RepoLoader.load_repo(file_path)
    docs = RepoLoader.load_documents(repo_path)
    print(f" → Loaded {len(docs)} documents")

    # 2️⃣ Extract code sections
    sections = CodeExtractor.split_documents_by_language(docs)
    print(f" → Extracted {len(sections)} code sections")

    # 3️⃣ Analyze repo (RepoIntel)
    repo_analysis = RepoIntel.generate_report_from_documents(sections)
    print(" → Repo Intel Analysis:\n\n\n")
    print(repo_analysis, "\n\n\n")
    for key, value in repo_analysis.items():
        print(f" {key}: {value}")

    # 4️⃣ Generate embeddings
    emb_model = EmbeddingStoreFAISS()
    emb_model.add_sections(sections)
    # print(f" → Created {len(vectors)} embeddings")

    # 5️⃣ Initialize FAISS search
    emb_model.save()
    faiss_store = emb_model.load()
    print("----------------", faiss_store)
    search_index = CodeSearchFAISS(faiss_store=faiss_store)
    # search_index.build_index(vectors, metadata)
    # print(" → FAISS index built")

    # 6️⃣ Perform search query
    # results = search_index.query(query)
    # print(f"\nSearch results for query: '{query}'")
    # print("*********************", results)
    # for i in results:
    #     print(i)
    #     for j in i:
    #         print(j)
    # for i, res in enumerate(results[:5]):
    #     print(f"{i+1}. File: {res['file']}, Lines: {res['start_line']}-{res['end_line']}")
    #     print(res['content'][:200], "...") # print first 200 chars
    #     print("-----")

def process_repository_for_graphflow(file_path: str, project_id: str = None, status_callback=None):
    """
    NEW FUNCTION: Process repository and prepare it for GraphFlow analysis.
    
    This wraps the existing process_repository() function and adds GraphFlow integration.
    
    Args:
        file_path: Path to ZIP file or GitHub URL
        project_id: Optional project ID (generates UUID if not provided)
        status_callback: Optional callback function for progress updates
    
    Returns:
        tuple: (project_id, project_dir_path)
        
    Example:
        project_id, project_dir = process_repository_for_graphflow("repo.zip")
        
        # Then call GraphFlow endpoint:
        import requests
        response = requests.post(
            f"http://127.0.0.1:8000/projects/{project_id}/analyze/graphflow"
        )
    """
    
    # Generate project_id if not provided
    if not project_id:
        project_id = str(uuid.uuid4())
    
    print(f"\n{'='*80}")
    print(f"🔬 Processing repository for GraphFlow: {file_path}")
    print(f"📋 Project ID: {project_id}")
    print(f"{'='*80}\n")
    
    try:
        # 1️⃣ Load repo and documents
        if status_callback: status_callback("Loading repository files...")
        print("1️⃣ Loading repository...")
        repo_path = RepoLoader.load_repo(file_path)
        docs = RepoLoader.load_documents(repo_path)
        print(f"   ✓ Loaded {len(docs)} documents\n")
        
        if not docs:
            raise ValueError("No documents found in repository")

        # 2️⃣ Extract code sections
        if status_callback: status_callback("Extracting code sections...")
        print("2️⃣ Extracting code sections...")
        sections = CodeExtractor.split_documents_by_language(docs)
        print(f"   ✓ Extracted {len(sections)} code sections\n")
        
        if not sections:
            raise ValueError("No code sections extracted")

        # 3️⃣ Analyze repo (RepoIntel)
        if status_callback: status_callback("Analyzing repository structure...")
        print("3️⃣ Analyzing repository with LLM...")
        try:
            repo_analysis = RepoIntel.generate_report_from_documents(sections)
            print("   ✓ Repo Analysis:")
            for key, value in repo_analysis.items():
                print(f"     - {key}: {value}")
            print()
        except Exception as e:
            print(f"   ⚠️  RepoIntel failed: {str(e)}")
            print("   ℹ️  Using fallback analysis\n")
            repo_analysis = {
                "stack": "Unknown",
                "framework": "None",
                "entry_points": []
            }

        # 4️⃣ Generate embeddings
        if status_callback: status_callback("Generating embeddings...")
        print("4️⃣ Generating embeddings...")
        emb_model = EmbeddingStoreFAISS()
        emb_model.add_sections(sections)
        print(f"   ✓ Created embeddings for {len(sections)} sections\n")

        # 5️⃣ Save in GraphFlow format
        if status_callback: status_callback("Saving project data...")
        print("5️⃣ Saving for GraphFlow integration...")
        project_dir = save_for_graphflow(
            project_id=project_id,
            repo_analysis=repo_analysis,
            sections_count=len(sections),
            faiss_store=emb_model.store
        )
        
        print(f"\n{'='*80}")
        print("✅ SUCCESS! Repository ready for GraphFlow analysis")
        print(f"{'='*80}\n")
        print(f"📍 Project Location: {project_dir}")
        print(f"🔑 Project ID: {project_id}\n")
        print("🚀 Next Step: Call GraphFlow endpoint")
        print(f"   POST http://127.0.0.1:8000/projects/{project_id}/analyze/graphflow\n")
        
        return project_id, project_dir
        
    except Exception as e:
        print(f"\n❌ Processing Failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage - ORIGINAL (unchanged)
    # random_file = "C:\\Users\\dagarwal17\\Downloads\\1. Autogen Basics.zip"
    # test_query = "Tell about autogen agents"
    # process_repository(random_file, test_query)
    
    # NEW: GraphFlow Integration Example
    zip_file = "C:\\path\\to\\your\\repo.zip"  # CHANGE THIS
    
    # Process for GraphFlow
    project_id, project_dir = process_repository_for_graphflow(zip_file)
    
    print("\n" + "="*80)
    print("To run GraphFlow analysis, call:")
    print(f"POST http://127.0.0.1:8000/projects/{project_id}/analyze/graphflow")
    print("="*80 + "\n")