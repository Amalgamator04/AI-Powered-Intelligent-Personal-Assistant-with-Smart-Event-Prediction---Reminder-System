#!/usr/bin/env python
"""Quick verification script to check project structure"""
import sys

def verify_imports():
    """Verify all main imports work"""
    try:
        from config import Config
        print("[OK] config.py")
        
        from agent import PersonalAgent
        print("[OK] agent package")
        
        from database import VectorStore, SessionManager
        print("[OK] database package")
        
        from processing import TextProcessor
        print("[OK] processing package")
        
        from helper import voice_search, append_to_kb
        print("[OK] helper package")
        
        from ollama_runner import OllamaClient
        print("[OK] ollama_runner.py")
        
        print("\n[SUCCESS] All imports successful! Project structure is correct.")
        return True
    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        return False

if __name__ == '__main__':
    success = verify_imports()
    sys.exit(0 if success else 1)
