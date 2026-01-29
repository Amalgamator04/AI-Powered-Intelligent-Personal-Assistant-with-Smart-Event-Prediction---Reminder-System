# AI-Powered Intelligent Personal Assistant with Smart Event Prediction & Reminder System

An intelligent personal assistant that uses AI to manage your knowledge base, answer questions, and provide smart reminders. Built with Ollama, ChromaDB, and speech recognition.

## Features

- 📚 **Knowledge Base Management**: Add, store, and retrieve information from your personal knowledge base
- 💬 **Interactive Chat**: Chat with your assistant using natural language
- 🎤 **Voice Input**: Use voice commands to interact with the assistant
- 🔍 **Semantic Search**: Find relevant information using vector embeddings
- 📊 **Session Management**: Track conversations and maintain context
- 🤖 **AI-Powered**: Uses Ollama for LLM and embeddings

## Prerequisites

- Python 3.8+
- Ollama installed and running locally (default: http://localhost:11434)
- Required Ollama models:
  - Embedding model: `granite-embedding:30m` (or update in `config.py`)
  - LLM model: `mistral` or `ministral-3:3b` (or update in `config.py`)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-Powered-Intelligent-Personal-Assistant-with-Smart-Event-Prediction---Reminder-System
```

2. Create a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama models (if not already installed):
```bash
ollama pull granite-embedding:30m
ollama pull mistral
# or
ollama pull ministral-3:3b
```

## Project Structure

```
.
├── agent/                 # AI agent implementation
│   ├── __init__.py
│   └── personal_agent.py
├── database/              # Database modules
│   ├── __init__.py
│   ├── session_manager.py
│   └── vector_store.py
├── helper/                # Helper utilities
│   ├── __init__.py
│   ├── knowledge_base.py
│   └── speechtotext.py
├── processing/           # Text processing
│   ├── __init__.py
│   └── text_processor.py
├── config.py              # Configuration
├── main.py                # Main entry point
├── ollama_runner.py       # Ollama API client
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Usage

### Command Line Interface

#### Add to Knowledge Base
```bash
# Add text directly
python main.py --mode add --text "Text here"

# Add from file
python main.py --mode add --file path/to/file.txt

# Add using voice input
python main.py --mode add --input-type voice
```

#### Query Knowledge Base
```bash
# Text query
python main.py --mode query --text "Your question"

# Voice query
python main.py --mode query --input-type voice
```

#### Interactive Chat Mode
```bash
# Text chat
python main.py --mode chat

# Voice chat
python main.py --mode chat --input-type voice

# With custom model and temperature
python main.py --mode chat --model mistral --temperature 0.8
```

#### View Statistics
```bash
python main.py --mode stats
```

### Programmatic Usage

```python
from config import Config
from database.vector_store import VectorStore
from database.session_manager import SessionManager
from agent.personal_agent import PersonalAgent

# Initialize
Config.create_dirs()
vector_store = VectorStore(str(Config.VECTOR_DB_PATH))
session_manager = SessionManager(str(Config.METADATA_DB_PATH))
agent = PersonalAgent(vector_store, session_manager)

# Add knowledge
agent.add_to_knowledge_base("Your text here", source="manual")

# Query
result = agent.query("Your question")

# Chat
session_id = agent.start_session()
response = agent.chat("Hello!")
```

## Configuration

Edit `config.py` to customize:
- Database paths
- Ollama base URL
- Embedding and LLM models
- Chunk size and overlap
- Temperature and other LLM settings

## Testing

Run the test suite:
```bash
python test_agent.py
```

## Troubleshooting

1. **Ollama connection errors**: Ensure Ollama is running on `http://localhost:11434`
2. **Model not found**: Pull the required models using `ollama pull <model-name>`
3. **Speech recognition issues**: Install PyAudio and ensure microphone permissions
4. **Import errors**: Ensure all `__init__.py` files are present in package directories

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
