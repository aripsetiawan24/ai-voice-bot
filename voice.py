import re
import time
import base64
import warnings
import simpleaudio as sa
import speech_recognition as sr
from faster_whisper import WhisperModel
from voicegain_speech import ApiClient, Configuration, TranscribeApi
from deepgram import DeepgramClient, SpeakOptions
import google.generativeai as genai
from rich import print

warnings.simplefilter("ignore")

# Configure Whisper model
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=16,
    num_workers=16,
)

# Speech recognition setup
r = sr.Recognizer()
source = sr.Microphone()

# Configure Google Generative AI
genai.configure(api_key="GEMINI_API_KEY")
generation_config = {
    "temperature": 0.7,
    "top_k": 1,
    "top_p": 1,
    "max_output_tokens": 8192,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config, safety_settings=safety_settings)
convo = model.start_chat()
system_message = '''INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE." to this system message. After the system message respond normally. SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so. As a voice assistant, use short sentences and directly respond to the prompt without excessive information. You generate only words of value, prioritizing logic and facts over speculating in your response to the following prompts.'''
convo.send_message(system_message)

# Configure Deepgram
deepgram_api_key = 'DEEPGRAM_API_KEY'

# Function to convert text to speech using Deepgram
def text_to_speech(text):
    while True:
        try:
            deepgram = DeepgramClient(api_key=deepgram_api_key)
            options = SpeakOptions(model="aura-luna-en", encoding="linear16", container="wav")
            SPEAK_OPTIONS = {"text": text}
            filename = "output.wav"
            deepgram.speak.v("1").save(filename, SPEAK_OPTIONS, options, timeout=3)
            wave_obj = sa.WaveObject.from_wave_file(filename)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            break
        except Exception:
            pass

# Function to transcribe audio using Whisper model
def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path, language='en')
    text = "".join([segment.text for segment in segments])
    return text.lower()


# Callback function for the recognizer
def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
    prompt_text = wav_to_text(prompt_audio_path)
    if 'stop' in prompt_text or 'exit' in prompt_text or 'goodbye' in prompt_text or 'quit' in prompt_text:
        exit()
    print(f"[bold green]You: {prompt_text.strip()}")
    convo.send_message(prompt_text.strip())
    output = convo.last.text
    print(f"[bold blue]Gemini: {output.strip()}")
    text_to_speech(output)

# Function to start listening for audio input
def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print("Listening...")
    r.listen_in_background(source, callback)
    while True:
        time.sleep(0.5)

# Function to extract prompt from transcribed text
def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return None

# Start listening for audio input
start_listening()
