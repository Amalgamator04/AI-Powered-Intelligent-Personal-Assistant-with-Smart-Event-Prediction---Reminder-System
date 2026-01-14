import speech_recognition as sr
import webbrowser as wb

def voice_search():
    chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print('Say something...')
        audio = r.listen(source)
        print('Done!')

    try:
        text = r.recognize_google(audio)
        print('Google thinks you said:', text)
        search_url = 'https://www.google.com/search?q=' + text
        try:
            wb.get(chrome_path).open(search_url)
        except Exception:
            # non-fatal if browser can't be opened
            pass
        return text

    except Exception as e:
        print('Error:', e)
        return None

# Note: do not auto-run the function on import; caller should invoke `voice_search()`.
