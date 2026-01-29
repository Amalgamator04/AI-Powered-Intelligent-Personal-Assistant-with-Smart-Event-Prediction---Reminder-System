# view_knowledge_base.py
"""View all documents stored in the knowledge base"""
from config import Config
from database.vector_store import VectorStore

def main():
    Config.create_dirs()
    vector_store = VectorStore(str(Config.VECTOR_DB_PATH))
    
    # Get all documents from the collection
    collection = vector_store.collection
    count = collection.count()
    
    print(f"Total documents in knowledge base: {count}\n")
    
    if count > 0:
        # Get all documents
        # ChromaDB's get() method can retrieve all documents
        try:
            results = collection.get()
            
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])
            ids = results.get('ids', [])
            
            print("=" * 80)
            print("KNOWLEDGE BASE CONTENTS")
            print("=" * 80)
            
            for i, (doc_id, doc, meta) in enumerate(zip(ids, documents, metadatas), 1):
                print(f"\n[{i}] ID: {doc_id}")
                if meta:
                    print(f"Timestamp: {meta.get('timestamp', 'N/A')}")
                    print(f"Source: {meta.get('source', 'N/A')}")
                    if 'chunk_index' in meta:
                        print(f"Chunk {meta.get('chunk_index', 'N/A')} of {meta.get('total_chunks', 'N/A')}")
                print(f"\nText:")
                print(f"{doc}")
                print("-" * 80)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            print("\nTrying alternative method...")
            # Alternative: query with a very general query to get all documents
            try:
                # Get a sample query to retrieve documents
                results = collection.get(limit=count)
                documents = results.get('documents', [])
                metadatas = results.get('metadatas', [])
                ids = results.get('ids', [])
                
                for i, (doc_id, doc, meta) in enumerate(zip(ids, documents, metadatas), 1):
                    print(f"\n[{i}] ID: {doc_id}")
                    if meta:
                        print(f"Timestamp: {meta.get('timestamp', 'N/A')}")
                        print(f"Source: {meta.get('source', 'N/A')}")
                    print(f"\nText: {doc[:500]}{'...' if len(doc) > 500 else ''}")
                    print("-" * 80)
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")
    else:
        print("No documents found in knowledge base.")
        print("Add some text using: python main.py --mode add --text 'Your text here'")

if __name__ == '__main__':
    main()
