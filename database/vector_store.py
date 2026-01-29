# database/vector_store.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid
from datetime import datetime
from ollama_runner import OllamaClient
from config import Config

class VectorStore:
    def __init__(self, persist_directory: str, collection_name: str = "knowledge_base",
                 embedding_model: str = None):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        # Use Ollama for embeddings
        self.ollama_client = OllamaClient(base_url=Config.OLLAMA_BASE_URL)
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using Ollama"""
        embeddings = []
        for text in texts:
            embedding = self.ollama_client.get_embeddings(
                model=self.embedding_model,
                prompt=text
            )
            embeddings.append(embedding)
        return embeddings
    
    def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to vector store"""
        if not texts:
            return
        
        print(f"Generating embeddings for {len(texts)} documents...")
        # Generate embeddings using Ollama
        embeddings = self.get_embeddings(texts)
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Add timestamp to metadata and clean None values (ChromaDB doesn't accept None)
        if metadata is None:
            metadata = [{} for _ in texts]
        
        # Clean metadata: remove None values (ChromaDB doesn't accept None)
        cleaned_metadata = []
        for meta in metadata:
            cleaned_meta = {k: v for k, v in meta.items() if v is not None}
            cleaned_meta['timestamp'] = datetime.now().isoformat()
            cleaned_metadata.append(cleaned_meta)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=cleaned_metadata,
            ids=ids
        )
        
        print(f"Successfully added {len(texts)} documents to vector store.")
        return ids
    
    def search(self, query: str, n_results: int = 5, filter_dict: Dict = None):
        """Search for similar documents"""
        print(f"Searching for: {query}")
        
        # Generate query embedding using Ollama
        query_embedding = self.ollama_client.get_embeddings(
            model=self.embedding_model,
            prompt=query
        )
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        return results
    
    def delete_by_ids(self, ids: List[str]):
        """Delete documents by IDs"""
        self.collection.delete(ids=ids)
        print(f"Deleted {len(ids)} documents.")
    
    def update_documents(self, ids: List[str], texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Update existing documents in the collection"""
        if not ids or not texts:
            return
        
        if len(ids) != len(texts):
            raise ValueError("Number of IDs must match number of texts")
        
        print(f"Updating {len(ids)} documents...")
        
        # Generate new embeddings
        embeddings = self.get_embeddings(texts)
        
        # Prepare metadata
        if metadata is None:
            metadata = [{} for _ in texts]
        
        # Clean metadata
        cleaned_metadata = []
        for meta in metadata:
            cleaned_meta = {k: v for k, v in meta.items() if v is not None}
            cleaned_meta['timestamp'] = datetime.now().isoformat()
            cleaned_meta['updated'] = True
            cleaned_metadata.append(cleaned_meta)
        
        # Delete old documents and add new ones with same IDs
        self.collection.delete(ids=ids)
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=cleaned_metadata,
            ids=ids
        )
        
        print(f"Successfully updated {len(ids)} documents.")
        return ids
    
    def get_documents_by_ids(self, ids: List[str]) -> Dict:
        """Get documents by their IDs"""
        if not ids:
            return {'documents': [], 'metadatas': [], 'ids': []}
        
        results = self.collection.get(ids=ids)
        return results
    
    def get_collection_stats(self):
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'embedding_model': self.embedding_model,
            'llm_model': Config.LLM_MODEL
        }
    
    def reset_collection(self):
        """Delete all documents from the collection"""
        # Get all document IDs
        all_documents = self.collection.get()
        if all_documents and all_documents['ids']:
            self.collection.delete(ids=all_documents['ids'])
            print(f"Deleted {len(all_documents['ids'])} documents from vector store.")
            return len(all_documents['ids'])
        else:
            print("No documents to delete in vector store.")
            return 0