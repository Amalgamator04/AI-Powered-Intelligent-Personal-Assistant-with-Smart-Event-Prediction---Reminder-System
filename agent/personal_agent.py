# agent/personal_agent.py
from typing import List, Dict, Optional
import uuid
from datetime import datetime
from ollama_runner import OllamaClient

class PersonalAgent:
    def __init__(self, vector_store, session_manager, llm_model: str = 'mistral'):
        self.vector_store = vector_store
        self.session_manager = session_manager
        self.ollama_client = OllamaClient(base_url="http://localhost:11434")
        self.llm_model = llm_model
        self.current_session_id = None
    
    def start_session(self, metadata: Dict = None) -> str:
        """Start a new conversation session"""
        self.current_session_id = str(uuid.uuid4())
        self.session_manager.create_session(self.current_session_id, metadata)
        print(f"Started new session: {self.current_session_id}")
        return self.current_session_id
    
    def add_to_knowledge_base(self, text: str, source: str = 'manual', 
                              metadata: Dict = None):
        """Add text to knowledge base"""
        from processing.text_processor import TextProcessor
        from config import Config
        
        processor = TextProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = processor.chunk_text(text)
        
        print(f"Split text into {len(chunks)} chunks")
        
        # Prepare metadata
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            meta = metadata.copy() if metadata else {}
            meta.update({
                'source': source,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'original_text_length': len(text)
            })
            chunk_metadata.append(meta)
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(chunks, chunk_metadata)
        
        return doc_ids
    
    def reframe_query(self, user_query: str, temperature: float = 0.3) -> str:
        """
        Reframe the user query using LLM to add temporal context and improve RAG search.
        Adds relevant dates and formats the query for better semantic search.
        """
        # Get current date and time for context
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_day = now.strftime("%A")
        current_month = now.strftime("%B")
        current_year = now.strftime("%Y")
        
        # Create a prompt for query reframing
        reframe_prompt = f"""You are a query enhancement assistant. Your task is to reframe user queries to improve semantic search in a knowledge base.

Current Date Context: {current_date} ({current_day}, {current_month} {current_year})

User's Original Query: "{user_query}"

Instructions:
1. Add specific dates or temporal context if the query mentions relative time (e.g., "today", "last week", "next month")
2. Expand abbreviations and implicit references
3. Format the query in a clear, structured manner to improve semantic search
4. Include the current date context naturally if relevant
5. Keep the reframed query concise but comprehensive
6. Do NOT lose any key information from the original query

Provide ONLY the reframed query without any explanation or additional text."""
        
        try:
            print(f"Reframing query: {user_query}")
            reframed = self._call_ollama_llm(reframe_prompt, temperature=temperature)
            print(f"Reframed query: {reframed}")
            return reframed
        except Exception as e:
            print(f"Error reframing query: {e}. Using original query.")
            return user_query
    
    def query(self, question: str, n_results: int = None) -> Dict:
        """Query the knowledge base with query reframing for better RAG search"""
        from config import Config
        if n_results is None:
            n_results = Config.MAX_CONTEXT_CHUNKS
        
        # Reframe the query using LLM to add temporal context and improve search
        reframed_question = self.reframe_query(question)
        
        # Search vector store with reframed query
        results = self.vector_store.search(reframed_question, n_results=n_results)
        
        # Save user question to session (save original question)
        if self.current_session_id:
            self.session_manager.add_message(
                self.current_session_id, 
                'user', 
                question
            )
        
        # Extract relevant context
        context_chunks = results['documents'][0] if results['documents'] else []
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        distances = results.get('distances', [[]])[0] if results.get('distances') else []
        
        return {
            'question': question,
            'reframed_question': reframed_question,
            'context': context_chunks,
            'metadata': metadatas,
            'distances': distances
        }
    
    def chat(self, message: str, use_context: bool = True, temperature: float = 0.7) -> str:
        """Chat with the agent using knowledge base context"""
        from config import Config
        # Get relevant context from knowledge base
        query_result = self.query(message, n_results=Config.MAX_CONTEXT_CHUNKS) if use_context else {'context': []}
        
        # Get session history for context
        history = []
        if self.current_session_id:
            history = self.session_manager.get_session_history(
                self.current_session_id, 
                limit=10  # Last 5 exchanges
            )
        
        # Build the complete prompt
        prompt = self._build_prompt(message, query_result['context'], history)
        
        # Call Ollama LLM using generate
        print("Generating response...")
        response = self._call_ollama_llm(prompt, temperature)
        
        # Save assistant response to session
        if self.current_session_id:
            self.session_manager.add_message(
                self.current_session_id,
                'assistant',
                response
            )
        
        return response
    
    def _build_prompt(self, user_message: str, context_chunks: List[str], 
                      history: List[Dict]) -> str:
        """Build a comprehensive prompt with context and history"""
        prompt_parts = []
        
        # Get current date and time
        now = datetime.now()
        current_datetime = now.strftime("%A, %B %d, %Y at %I:%M %p")
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        day_of_week = now.strftime("%A")
        
        # System instruction with datetime context
        prompt_parts.append(
            f"Current Date and Time: {current_datetime} ({current_date} {current_time})\n"
            f"Day of Week: {day_of_week}\n\n"
            "You are a helpful personal AI assistant with access to the user's knowledge base. "
            "Your role is to answer questions based on the provided context from their notes, "
            "documents, and previous conversations.\n\n"
            "Guidelines:\n"
            "- If the context contains relevant information, use it to provide accurate answers.\n"
            "- If the context doesn't contain enough information, say so and provide general knowledge if appropriate.\n"
            "- Be conversational, helpful, and concise.\n"
            "- Reference specific information from the context when relevant.\n"
            "- Use the current date and time to understand time-sensitive queries and provide accurate temporal context.\n"
        )
        
        # Add conversation history if available
        if history:
            prompt_parts.append("\n--- Conversation History ---")
            for msg in history[-6:]:  # Last 3 exchanges
                role = "User" if msg['role'] == 'user' else "Assistant"
                prompt_parts.append(f"{role}: {msg['content']}")
            prompt_parts.append("--- End of History ---\n")
        
        # Add context from knowledge base
        if context_chunks:
            from config import Config
            prompt_parts.append("\n--- Relevant Information from Knowledge Base ---")
            # Use all context chunks (already limited by MAX_CONTEXT_CHUNKS in query)
            for i, chunk in enumerate(context_chunks[:Config.MAX_CONTEXT_CHUNKS], 1):
                prompt_parts.append(f"[Source {i}]: {chunk}")
            prompt_parts.append("--- End of Knowledge Base Context ---\n")
        
        # Add current user message
        prompt_parts.append(f"\nUser: {user_message}")
        prompt_parts.append("\nAssistant:")
        
        return "\n".join(prompt_parts)
    
    def _call_ollama_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call Ollama LLM using generate method"""
        try:
            response = self.ollama_client.generate(
                model=self.llm_model,
                prompt=prompt,
                temperature=temperature
            )
            return response.strip()
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return f"I encountered an error generating a response: {str(e)}"
    
    def simple_completion(self, prompt: str, temperature: float = 0.7) -> str:
        """Simple completion without context or history"""
        try:
            response = self.ollama_client.generate(
                model=self.llm_model,
                prompt=prompt,
                temperature=temperature
            )
            return response.strip()
        except Exception as e:
            print(f"Error: {e}")
            return f"Error: {str(e)}"
    
    def get_stats(self):
        """Get knowledge base statistics"""
        stats = self.vector_store.get_collection_stats()
        
        # Add session info if available
        if self.current_session_id:
            history = self.session_manager.get_session_history(self.current_session_id)
            stats['current_session'] = {
                'session_id': self.current_session_id,
                'message_count': len(history)
            }
        
        return stats
    
    def reset_all(self):
        """Reset all data: vector store and session database"""
        print("Resetting all data...")
        
        # Reset vector store
        vector_docs_deleted = self.vector_store.reset_collection()
        
        # Reset session database
        session_data = self.session_manager.reset_database()
        
        # Reset current session
        self.current_session_id = None
        
        return {
            'vector_documents_deleted': vector_docs_deleted,
            'sessions_deleted': session_data['sessions'],
            'messages_deleted': session_data['messages'],
            'status': 'success'
        }
    
    def detect_update_intent(self, user_input: str, temperature: float = 0.2) -> Dict:
        """
        Use LLM to detect if user input is meant to update existing knowledge.
        Returns: {'is_update': bool, 'topic': str, 'reason': str}
        """
        intent_prompt = f"""You are an intent detection assistant. Analyze the user input and determine if it's meant to UPDATE existing information or ADD new information.

User Input: "{user_input}"

Consider:
1. Words like "updated", "changed", "increased", "decreased", "corrected", "now", "since", "recently"
2. References to previously mentioned facts (e.g., "my rent is now X" suggests update to existing rent info)
3. Corrections or clarifications (e.g., "actually it's X" suggests update)

Respond in JSON format:
{{
    "is_update": true/false,
    "topic": "main topic or entity being updated (e.g., 'house_rent', 'work_location')",
    "reason": "brief explanation"
}}

If it seems like new information, set is_update to false."""

        try:
            response = self._call_ollama_llm(intent_prompt, temperature=temperature)
            import json
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                return {'is_update': False, 'topic': '', 'reason': 'Could not parse response'}
        except Exception as e:
            print(f"Error detecting update intent: {e}")
            return {'is_update': False, 'topic': '', 'reason': str(e)}
    
    def find_related_documents(self, query: str, n_results: int = 5) -> Dict:
        """
        Find related documents in the knowledge base for a given query.
        Returns documents that might need to be updated.
        """
        results = self.vector_store.search(query, n_results=n_results)
        return results
    
    def merge_knowledge(self, original_text: str, update_text: str, topic: str = "", temperature: float = 0.3) -> str:
        """
        Use LLM to intelligently merge old knowledge with new update.
        Returns the merged/updated knowledge.
        """
        merge_prompt = f"""You are a knowledge merge assistant. Merge the original knowledge with the new update information.

Topic: {topic}

Original Knowledge:
"{original_text}"

New Update Information:
"{update_text}"

Task:
1. Identify what has changed or been updated
2. Merge the new information with the old, keeping all relevant details
3. Remove or update outdated information
4. Create a coherent, up-to-date knowledge entry
5. Preserve context and relationships from the original

Provide ONLY the merged knowledge without any explanation or metadata."""

        try:
            merged = self._call_ollama_llm(merge_prompt, temperature=temperature)
            return merged
        except Exception as e:
            print(f"Error merging knowledge: {e}")
            return original_text
    
    def add_or_update_knowledge_base(self, text: str, source: str = 'manual', 
                                     metadata: Dict = None) -> Dict:
        """
        Intelligent method that decides whether to ADD new knowledge or UPDATE existing.
        Returns: {
            'action': 'added' or 'updated',
            'doc_ids': [...],
            'details': {...}
        }
        """
        from processing.text_processor import TextProcessor
        from config import Config
        
        print("Analyzing input for add or update...")
        
        # Step 1: Detect if this is an update
        intent = self.detect_update_intent(text)
        print(f"Intent detection: {intent}")
        
        details = {
            'intent': intent,
            'related_documents': None,
            'merged': False
        }
        
        # Step 2: If not detected as update, just add it
        if not intent.get('is_update', False):
            print("Adding as new knowledge...")
            doc_ids = self.add_to_knowledge_base(text, source=source, metadata=metadata)
            return {
                'action': 'added',
                'doc_ids': doc_ids,
                'details': details
            }
        
        # Step 3: If update, find related documents
        print(f"Update detected for topic: {intent.get('topic', 'unknown')}")
        related_docs = self.find_related_documents(text, n_results=3)
        details['related_documents'] = related_docs
        
        # Step 4: If related documents found, merge and update
        if related_docs.get('documents') and related_docs['documents'][0]:
            # Get ALL chunks that belong to the most relevant document
            all_related_ids = related_docs['ids'][0]  # All IDs for the most relevant document
            all_related_docs = related_docs['documents'][0]  # All document chunks
            most_relevant_doc = all_related_docs[0]  # Use first chunk as representative
            
            if all_related_ids and most_relevant_doc:
                print(f"Found related document with {len(all_related_ids)} chunks, merging knowledge...")
                
                # Merge the knowledge
                merged_text = self.merge_knowledge(
                    most_relevant_doc, 
                    text, 
                    topic=intent.get('topic', '')
                )
                details['merged'] = True
                details['original_doc'] = most_relevant_doc
                details['merged_doc'] = merged_text
                
                # Process the merged text into new chunks
                processor = TextProcessor(
                    chunk_size=Config.CHUNK_SIZE,
                    chunk_overlap=Config.CHUNK_OVERLAP
                )
                new_chunks = processor.chunk_text(merged_text)
                
                # Prepare metadata for updated chunks
                new_chunk_metadata = []
                for i, chunk in enumerate(new_chunks):
                    meta = metadata.copy() if metadata else {}
                    meta.update({
                        'source': source,
                        'chunk_index': i,
                        'total_chunks': len(new_chunks),
                        'original_text_length': len(merged_text),
                        'is_update': True,
                        'updated_topic': intent.get('topic', '')
                    })
                    new_chunk_metadata.append(meta)
                
                # Delete all old chunks and add new ones with same number of chunks
                # If new chunks are different count, use only the matching count
                ids_to_update = all_related_ids[:len(new_chunks)]
                
                # If we have fewer new chunks, delete the extra old chunks
                if len(new_chunks) < len(all_related_ids):
                    ids_to_delete = all_related_ids[len(new_chunks):]
                    self.vector_store.delete_by_ids(ids_to_delete)
                
                # Update the documents
                doc_ids = self.vector_store.update_documents(
                    ids_to_update,
                    new_chunks[:len(ids_to_update)],
                    new_chunk_metadata[:len(ids_to_update)]
                )
                
                return {
                    'action': 'updated',
                    'doc_ids': doc_ids,
                    'details': details
                }
        
        # Step 5: If no related documents found but intent was update, add as new
        print("No related documents found, adding as new...")
        doc_ids = self.add_to_knowledge_base(text, source=source, metadata=metadata)
        return {
            'action': 'added',
            'doc_ids': doc_ids,
            'details': details
        }
