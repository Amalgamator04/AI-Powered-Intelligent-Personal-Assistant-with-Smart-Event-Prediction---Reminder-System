# application.py
# Simple voice-to-knowledge-base entry point
from helper.speechtotext import voice_search
from helper.knowledge_base import append_to_kb

def main():
    """Simple application: listen to voice and append to knowledge base"""
    voice_text = voice_search()
    if voice_text:
        append_to_kb(voice_text)
    else:
        print('No speech recognized; nothing appended.')

if __name__ == '__main__':
    main()