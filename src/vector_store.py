"""
ChromaDB vector store for semantic search with subject filtering
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from . import config


class VectorStore:
    """ChromaDB-based vector store with subject metadata filtering"""
    
    def __init__(self):
        """Initialize ChromaDB client and embedding model"""
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model (supports Arabic and English)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("Embedding model loaded successfully")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"description": "Thanaweya Amma educational content"}
        )
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to ChromaDB with embeddings
        
        Args:
            documents: List of document dicts with 'text' and 'metadata' keys
        """
        if not documents:
            print("No documents to add")
            return
        
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        
        # Create unique IDs
        ids = [f"{meta['subject']}_{meta['filename']}_{meta['chunk_id']}" 
               for meta in metadatas]
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        print(f"Added {len(documents)} documents to ChromaDB")
    
    def semantic_search(
        self,
        query: str,
        subject: str,
        top_k: int = config.TOP_K_RETRIEVAL
    ) -> List[Dict]:
        """
        Perform semantic search filtered by subject
        
        Args:
            query: Search query
            subject: Subject to filter by
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata and scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).tolist()[0]
        
        # Search with subject filter
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"subject": subject}  # Filter by subject metadata
        )
        
        # Format results
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                documents.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "score": 1 / (1 + results["distances"][0][i])  # Convert distance to similarity score
                })
        
        return documents
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": config.CHROMA_COLLECTION_NAME
        }
    
    def reset_collection(self):
        """Reset (delete) the collection - use with caution!"""
        self.client.delete_collection(config.CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"description": "Thanaweya Amma educational content"}
        )
        print("Collection reset successfully")


if __name__ == "__main__":
    # Test the vector store
    vector_store = VectorStore()
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")
