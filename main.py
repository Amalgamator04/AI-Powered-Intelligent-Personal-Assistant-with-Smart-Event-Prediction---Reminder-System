# main.py
from config import Config
from database.vector_store import VectorStore
from database.session_manager import SessionManager
from agent.personal_agent import PersonalAgent
from helper.speechtotext import voice_search
import argparse

def main():
    # Initialize config
    Config.create_dirs()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Personal AI Knowledge Base Agent')
    parser.add_argument('--mode', choices=['add', 'query', 'chat', 'stats'], 
                       required=True, help='Operation mode')
    parser.add_argument('--input-type', choices=['text', 'voice'], 
                       default='text', help='Input type')
    parser.add_argument('--text', type=str, help='Text input')
    parser.add_argument('--source', type=str, default='manual', 
                       help='Source of the knowledge')
    parser.add_argument('--file', type=str, help='File path to add to knowledge base')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='LLM temperature (0.0-1.0)')
    parser.add_argument('--model', type=str, default=Config.LLM_MODEL,
                       help='Ollama model to use')
    args = parser.parse_args()
    
    # Initialize components
    print("Initializing Personal AI Agent...")
    vector_store = VectorStore(str(Config.VECTOR_DB_PATH))
    session_manager = SessionManager(str(Config.METADATA_DB_PATH))
    agent = PersonalAgent(vector_store, session_manager, llm_model=args.model)
    
    if args.mode == 'stats':
        # Show statistics
        stats = agent.get_stats()
        print("\n=== Knowledge Base Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        return
    
    if args.mode == 'add':
        # Add to knowledge base
        text = None
        
        if args.file:
            # Read from file
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Read {len(text)} characters from {args.file}")
        elif args.input_type == 'voice':
            print("🎤 Listening for speech...")
            text = voice_search()
            if not text:
                print("❌ No speech recognized.")
                return
            print(f"✓ Recognized: {text}")
        else:
            text = args.text or input("Enter text to add: ")
        
        if text:
            print(f"\n📝 Adding to knowledge base...")
            doc_ids = agent.add_to_knowledge_base(
                text, 
                source=args.source,
                metadata={
                    'input_type': args.input_type,
                    'file': args.file if args.file else None
                }
            )
            print(f"✓ Successfully added {len(doc_ids)} chunks to knowledge base.")
    
    elif args.mode == 'query':
        # Query knowledge base
        if args.input_type == 'voice':
            print("🎤 Listening for question...")
            question = voice_search()
            if not question:
                print("❌ No speech recognized.")
                return
            print(f"Question: {question}")
        else:
            question = args.text or input("Enter your question: ")
        
        result = agent.query(question)
        print(f"\n❓ Question: {result['question']}")
        print(f"\n📚 Found {len(result['context'])} relevant chunks:\n")
        
        for i, (chunk, distance) in enumerate(zip(result['context'], result['distances']), 1):
            print(f"\n[{i}] (similarity: {1 - distance:.3f})")
            print(f"{chunk[:300]}{'...' if len(chunk) > 300 else ''}")
            print("-" * 80)
    
    elif args.mode == 'chat':
        # Interactive chat mode
        session_id = agent.start_session({'mode': 'chat', 'input_type': args.input_type})
        print(f"\n💬 Chat Mode Started")
        print(f"Session ID: {session_id}")
        print(f"Model: {args.model}")
        print(f"Temperature: {args.temperature}")
        print("\nCommands:")
        print("  - Type your question to chat")
        print("  - Type 'exit', 'quit', or 'bye' to end")
        print("  - Type 'stats' to see knowledge base info")
        print("\n" + "=" * 80)
        
        while True:
            if args.input_type == 'voice':
                print("\n🎤 Listening...")
                user_input = voice_search()
                if not user_input:
                    print("❌ No speech recognized, try again.")
                    continue
                print(f"\n🧑 You: {user_input}")
            else:
                user_input = input("\n🧑 You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = agent.get_stats()
                print("\n📊 Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            response = agent.chat(user_input)
            print(f"\n🤖 Assistant: {response}")
            print("-" * 80)

if __name__ == '__main__':
    main()