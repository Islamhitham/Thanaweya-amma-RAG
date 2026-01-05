"""
Hybrid retrieval combining BM25 (lexical) and semantic search
"""
from rank_bm25 import BM25Okapi
from typing import List, Dict
import numpy as np
from collections import defaultdict
from . import config


class HybridRetriever:
    """Hybrid retriever combining BM25 and semantic search"""
    
    def __init__(self, vector_store):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: VectorStore instance for semantic search
        """
        self.vector_store = vector_store
        self.bm25_indices = {}  # Subject -> BM25 index
        self.corpus_docs = {}   # Subject -> List of documents
    
    def build_bm25_index(self, subject: str, documents: List[Dict]):
        """
        Build BM25 index for a specific subject
        
        Args:
            subject: Subject name
            documents: List of document dicts with 'text' and 'metadata'
        """
        if not documents:
            print(f"No documents to index for {subject}")
            return
        
        # Tokenize documents (simple whitespace tokenization)
        tokenized_corpus = [doc["text"].lower().split() for doc in documents]
        
        # Build BM25 index
        self.bm25_indices[subject] = BM25Okapi(tokenized_corpus)
        self.corpus_docs[subject] = documents
        
        print(f"Built BM25 index for {subject}: {len(documents)} documents")
    
    def bm25_search(self, query: str, subject: str, top_k: int) -> List[Dict]:
        """
        Perform BM25 search for a subject
        
        Args:
            query: Search query
            subject: Subject to search in
            top_k: Number of results to return
            
        Returns:
            List of documents with BM25 scores
        """
        if subject not in self.bm25_indices:
            print(f"Warning: No BM25 index for {subject}")
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        bm25_scores = self.bm25_indices[subject].get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # Only include non-zero scores
                doc = self.corpus_docs[subject][idx].copy()
                doc["bm25_score"] = float(bm25_scores[idx])
                results.append(doc)
        
        return results
    
    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict],
        semantic_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Combine BM25 and semantic results using Reciprocal Rank Fusion (RRF)
        
        Args:
            bm25_results: Results from BM25 search
            semantic_results: Results from semantic search
            k: RRF constant (default: 60)
            
        Returns:
            Fused and ranked results
        """
        # Create a score map: doc_text -> RRF score
        rrf_scores = defaultdict(float)
        doc_map = {}  # Store full document info
        
        # Add BM25 scores (weighted)
        for rank, doc in enumerate(bm25_results, 1):
            doc_text = doc["text"]
            rrf_scores[doc_text] += config.HYBRID_SEARCH_WEIGHTS["bm25"] * (1 / (k + rank))
            doc_map[doc_text] = doc
        
        # Add semantic scores (weighted)
        for rank, doc in enumerate(semantic_results, 1):
            doc_text = doc["text"]
            rrf_scores[doc_text] += config.HYBRID_SEARCH_WEIGHTS["semantic"] * (1 / (k + rank))
            if doc_text not in doc_map:
                doc_map[doc_text] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Format results
        results = []
        for doc_text, score in sorted_docs:
            doc = doc_map[doc_text].copy()
            doc["hybrid_score"] = score
            results.append(doc)
        
        return results
    
    def search(
        self,
        query: str,
        subject: str,
        top_k: int = config.TOP_K_RETRIEVAL
    ) -> List[Dict]:
        """
        Perform hybrid search (BM25 + semantic)
        
        Args:
            query: Search query
            subject: Subject to filter by
            top_k: Number of final results to return
            
        Returns:
            List of top-k documents ranked by hybrid score
        """
        # Retrieve more results for fusion
        retrieve_k = top_k * 2
        
        # Get BM25 results
        bm25_results = self.bm25_search(query, subject, retrieve_k)
        
        # Get semantic results
        semantic_results = self.vector_store.semantic_search(
            query, subject, retrieve_k
        )
        
        # Fuse results
        hybrid_results = self.reciprocal_rank_fusion(
            bm25_results,
            semantic_results
        )
        
        # Return top-k
        return hybrid_results[:top_k]


if __name__ == "__main__":
    # Testing code would go here
    print("Hybrid retriever module loaded")
