# ollama_runner.py
import requests
from typing import List, Optional

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
    
    def generate(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Ollama's generate endpoint"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {e}")
    
    def get_embeddings(self, model: str, prompt: str) -> List[float]:
        """Get embeddings for a text using Ollama's embeddings endpoint"""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model,
            "prompt": prompt
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error getting embeddings from Ollama: {e}")
