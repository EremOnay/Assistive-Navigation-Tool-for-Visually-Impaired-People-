import threading
import time
import os
from gtts import gTTS
from collections import deque

# ——————————————————————
# 1. TTS deque & worker
# ——————————————————————
tts_deque = deque(maxlen=1)
alert_playing = threading.Event()
tts_lock = threading.Lock()
done = 1

def _tts_worker():
    global done
    while True:
        # Wait until there is something in the deque
        while not tts_deque:
            time.sleep(0.001)
        with tts_lock:
            text, is_alert = tts_deque.popleft()
        done = 0
        if is_alert:
            alert_playing.set()
        start = time.perf_counter()
        tts = gTTS(text=text, lang='en')
        filename = "tts_output.mp3"
        tts.save(filename)

        inter = time.time()
        os.system("ffmpeg -y -i tts_output.mp3 -filter:a \"atempo=1.25\" fast_tts.mp3")
        print(f"encoding time: {time.time() - inter:.3f}s")

        elapsed = time.perf_counter() - start
        print(f'[gTTS] "{text}" generated in {elapsed:.3f}s')
        #os.system("mpg123 -q tts_output.mp3")
        os.system("mpg123 -q fast_tts.mp3")
        done = 1
        if is_alert:
            alert_playing.clear()

# Start the worker thread
threading.Thread(target=_tts_worker, daemon=True).start()

def say(text: str, interrupt=False, is_alert=False):
    """
    Always keeps only the latest message in the deque.
    If is_alert=True and an alert is playing, ignore new alert.
    """ 
    global done
    with tts_lock:
        
        if is_alert and alert_playing.is_set():
            print("\nAlert is playing, new alert ignored:", text)
            return
        
        elif alert_playing.is_set():
            print("\nAlert is playing, new message ignored:", text)
            return

        if is_alert == False:
            if done == 0:
                print("\nTTS not ready, new message ignored:", text)
                return
        
                
        if interrupt:
            tts_deque.clear()
            os.system("pkill -f mpg123")
        tts_deque.clear()
        tts_deque.append((text, is_alert))
        