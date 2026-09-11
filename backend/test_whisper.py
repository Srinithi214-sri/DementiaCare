import whisper

model = whisper.load_model("base")
result = model.transcribe("harvard.wav")
print(result["text"])