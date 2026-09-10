import whisper
from sentence_transformers import SentenceTransformer, util
from datetime import datetime, timezone

whisper_model = whisper.load_model("base")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

recent_questions = []

DISTRESS_KEYWORDS = [
    "help", "scared", "afraid", "confused", "lost",
    "don't know where", "where am i", "who are you",
    "i want to go home", "hurts", "pain", "can't breathe"
]

def check_repetition(text, threshold=0.75):
    if not recent_questions:
        recent_questions.append(text)
        return False, None
    embeddings = embed_model.encode([text] + recent_questions)
    sims = util.cos_sim(embeddings[0], embeddings[1:])
    max_sim = sims.max().item()
    recent_questions.append(text)
    return max_sim > threshold, max_sim

def check_distress(text):
    text_lower = text.lower()
    matched = [kw for kw in DISTRESS_KEYWORDS if kw in text_lower]
    return len(matched) > 0, matched

def process_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    text = result["text"].strip()

    is_repeat, rep_score = check_repetition(text)
    is_distress, matched_keywords = check_distress(text)

    events = []

    if is_repeat:
        events.append({
            "event_type": "repetitive_question",
            "source": "speech",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": round(rep_score, 3) if rep_score else None,
            "metadata": {"transcript": text}
        })

    if is_distress:
        events.append({
            "event_type": "distress_detected",
            "source": "speech",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": 1.0,
            "metadata": {"transcript": text, "matched_keywords": matched_keywords}
        })

    return {"transcript": text, "events": events}

if __name__ == "__main__":
    output = process_audio("mic_test.wav")
    print(output)