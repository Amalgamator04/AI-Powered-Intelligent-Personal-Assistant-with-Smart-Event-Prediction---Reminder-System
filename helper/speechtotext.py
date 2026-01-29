# helper/speechtotext.py
import speech_recognition as sr

def voice_search():
    """Convert speech to text using Google Speech Recognition"""
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print('🎤 Listening...')
        # Adjust for ambient noise
        r.adjust_for_ambient_noise(source, duration=0.5)
        # Listen for minimum 10 seconds (timeout=10) and allow up to 30 seconds of speech
        audio = r.listen(source, timeout=10, phrase_time_limit=30)
        print('✓ Processing...')

    try:
        text = r.recognize_google(audio)
        print(f'✓ Recognized: {text}')
        return text
    except sr.UnknownValueError:
        print('❌ Could not understand audio')
        return None
    except sr.RequestError as e:
        print(f'❌ Error with speech recognition service: {e}')
        return None
    except Exception as e:
        print(f'❌ Error: {e}')
        return None
