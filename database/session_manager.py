# database/session_manager.py
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import json

class SessionManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize session database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    last_updated TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            ''')
            conn.commit()
    
    def create_session(self, session_id: str, metadata: Dict = None):
        """Create a new session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO sessions (session_id, created_at, last_updated, metadata)
                VALUES (?, ?, ?, ?)
            ''', (
                session_id,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                json.dumps(metadata or {})
            ))
            conn.commit()
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO messages (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (session_id, role, content, datetime.now().isoformat()))
            
            # Update session last_updated
            conn.execute('''
                UPDATE sessions SET last_updated = ? WHERE session_id = ?
            ''', (datetime.now().isoformat(), session_id))
            
            conn.commit()
    
    def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get session conversation history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT role, content, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, limit))
            
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    'role': row[0],
                    'content': row[1],
                    'timestamp': row[2]
                })
            
            return list(reversed(messages))
    
    def reset_database(self):
        """Delete all sessions and messages from the database"""
        with sqlite3.connect(self.db_path) as conn:
            # Get count before deletion
            cursor = conn.execute('SELECT COUNT(*) FROM messages')
            message_count = cursor.fetchone()[0]
            
            cursor = conn.execute('SELECT COUNT(*) FROM sessions')
            session_count = cursor.fetchone()[0]
            
            # Delete all messages and sessions
            conn.execute('DELETE FROM messages')
            conn.execute('DELETE FROM sessions')
            conn.commit()
            
            print(f"Deleted {message_count} messages and {session_count} sessions from database.")
            return {'messages': message_count, 'sessions': session_count}