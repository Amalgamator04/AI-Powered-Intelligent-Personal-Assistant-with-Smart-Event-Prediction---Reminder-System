# test_agent.py
from config import Config
from database.vector_store import VectorStore
from database.session_manager import SessionManager
from agent.personal_agent import PersonalAgent

def test_knowledge_base():
    """Test the complete knowledge base system"""
    
    # Initialize
    Config.create_dirs()
    
    print("=" * 80)
    print("Testing Personal AI Knowledge Base Agent")
    print("=" * 80)
    
    # Create components
    vector_store = VectorStore(str(Config.VECTOR_DB_PATH))
    session_manager = SessionManager(str(Config.METADATA_DB_PATH))
    agent = PersonalAgent(vector_store, session_manager)
    
    # Test 1: Add knowledge
    print("\n1. Adding knowledge to the database...")
    knowledge_items = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Deep learning uses multiple layers of neural networks to learn complex patterns.",
        "Natural language processing helps computers understand and generate human language."
    ]
    
    for i, text in enumerate(knowledge_items, 1):
        agent.add_to_knowledge_base(
            text, 
            source="test_data",
            metadata={'category': 'AI', 'item_number': i}
        )
    
    # Test 2: Query knowledge base
    print("\n2. Testing knowledge retrieval...")
    query = "What is machine learning?"
    result = agent.query(query, n_results=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(result['context'])} relevant results:")
    for i, (chunk, distance) in enumerate(zip(result['context'], result['distances']), 1):
        print(f"\n[{i}] Similarity: {1-distance:.3f}")
        print(f"Text: {chunk}")
    
    # Test 3: Chat with context
    print("\n3. Testing chat with context...")
    session_id = agent.start_session({'test': 'chat_test'})
    
    questions = [
        "Tell me about Python programming language",
        "How is it different from machine learning?",
        "What about neural networks?"
    ]
    
    for question in questions:
        print(f"\n🧑 User: {question}")
        response = agent.chat(question)
        print(f"🤖 Assistant: {response}")
    
    # Test 4: Stats
    print("\n4. Knowledge base statistics:")
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✓ All tests completed!")
    print("=" * 80)

if __name__ == '__main__':
    test_knowledge_base()