import win32com.client

# Initialize Windows speech engine
speaker = win32com.client.Dispatch("SAPI.SpVoice")

# List all voices installed on this Windows PC
voices = speaker.GetVoices()

print("=== CHECKING INSTALLED WINDOWS VOICES ===")
print(f"Total voices found: {voices.Count}\n")

for i in range(voices.Count):
    voice = voices.Item(i)
    desc = voice.GetDescription()
    print(f"[{i}] {desc}")

print("\n-------------------------------------------")
print("Speaking a multilingual test...")

# Default test in English
speaker.Speak("Testing multilingual support for SIGNOVA.")
print("Test completed.")
