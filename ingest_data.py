"""
Data ingestion script to load PDFs into ChromaDB
"""
import sys
from pathlib import Path
from tqdm import tqdm

from src import config
from src.data_processing import DocumentProcessor
from src.vector_store import VectorStore
from src.hybrid_retriever import HybridRetriever


def setup_data_directories():
    """Create data directories if they don't exist"""
    config.DATA_DIR.mkdir(exist_ok=True)
    
    for subject in config.SUBJECTS:
        subject_dir = config.DATA_DIR / subject
        subject_dir.mkdir(exist_ok=True)
    
    print(f"✓ Data directories created at: {config.DATA_DIR}")


def ingest_data():
    """Main ingestion pipeline"""
    print("="*60)
    print("Thanaweya Amma RAG - Data Ingestion")
    print("="*60)
    
    # Setup directories
    setup_data_directories()
    
    # Initialize components
    print("\n[1/4] Initializing document processor...")
    processor = DocumentProcessor()
    
    print("[2/4] Processing PDFs...")
    all_subject_chunks = processor.process_all_subjects()
    
    # Check if we have any data
    total_chunks = sum(len(chunks) for chunks in all_subject_chunks.values())
    if total_chunks == 0:
        print("\n" + "="*60)
        print("WARNING: No PDFs found!")
        print("="*60)
        print("Please place your PDF files in the following directories:")
        for subject in config.SUBJECTS:
            print(f"  - {config.DATA_DIR / subject}/")
        print("\nExample structure:")
        print("  data/")
        print("    ├── arabic/")
        print("    │   ├── arabic_book1.pdf")
        print("    │   └── arabic_book2.pdf")
        print("    ├── math/")
        print("    │   └── math_textbook.pdf")
        print("    └── ...")
        print("="*60)
        sys.exit(1)
    
    print(f"\n[3/4] Initializing vector store...")
    vector_store = VectorStore()
    
    # Option to reset existing collection
    stats = vector_store.get_collection_stats()
    if stats['total_documents'] > 0:
        response = input(f"\nCollection already has {stats['total_documents']} documents. Reset? (y/n): ")
        if response.lower() == 'y':
            print("Resetting collection...")
            vector_store.reset_collection()
    
    print("\n[4/4] Indexing documents in ChromaDB...")
    
    # Add all documents to vector store
    all_chunks = []
    for subject, chunks in all_subject_chunks.items():
        all_chunks.extend(chunks)
    
    if all_chunks:
        vector_store.add_documents(all_chunks)
    
    # Build BM25 indices for hybrid retrieval
    print("\n[Bonus] Building BM25 indices for hybrid search...")
    retriever = HybridRetriever(vector_store)
    
    for subject, chunks in all_subject_chunks.items():
        if chunks:
            retriever.build_bm25_index(subject, chunks)
    
    # Final summary
    print("\n" + "="*60)
    print("INGESTION COMPLETE!")
    print("="*60)
    print(f"Total documents indexed: {len(all_chunks)}")
    print("\nDocuments per subject:")
    for subject, chunks in all_subject_chunks.items():
        print(f"  - {subject.upper()}: {len(chunks)} chunks")
    print("\n" + "="*60)
    print("You can now run: python main.py")
    print("="*60)


if __name__ == "__main__":
    try:
        ingest_data()
    except KeyboardInterrupt:
        print("\n\nIngestion interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
