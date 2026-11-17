import os
from gtts import gTTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")

os.makedirs(AUDIO_DIR, exist_ok=True)

# Multilingual dictionary
MULTILINGUAL_SENTENCES = {
    "thank_you": {
        "en": "Thank you",
        "hi": "धन्यवाद",
        "kn": "ಧನ್ಯವಾದಗಳು"
    },
    "i_need_help": {
        "en": "I need help",
        "hi": "मुझे मदद चाहिए",
        "kn": "ನನಗೆ ಸಹಾಯ ಬೇಕು"
    },
    "i_need_water": {
        "en": "I need water",
        "hi": "मुझे पानी चाहिए",
        "kn": "ನನಗೆ ನೀರು ಬೇಕು"
    },
    "i_am_in_danger": {
        "en": "I am in danger",
        "hi": "मैं खतरे में हूँ",
        "kn": "ನಾನು ಅಪಾಯದಲ್ಲಿದ್ದೇನೆ"
    },
    "i_need_food": {
        "en": "I need food",
        "hi": "मुझे खाना चाहिए",
        "kn": "ನನಗೆ ಊಟ ಬೇಕು"
    }
}

print("=== GENERATING MULTILINGUAL AUDIO (SIGNOVA) ===")
print("Generating audio for English, Hindi, and Kannada...\n")

total = len(MULTILINGUAL_SENTENCES) * 3
count = 0

for sign_key, trans in MULTILINGUAL_SENTENCES.items():
    for lang, text in trans.items():
        count += 1
        filename = f"{sign_key}_{lang}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        print(f"[{count}/{total}] Generating ({lang}): \"{text}\" -> {filename}...")

        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(filepath)

print("\nAll multilingual audio files generated successfully!")
print(f"Saved in: {AUDIO_DIR}")
