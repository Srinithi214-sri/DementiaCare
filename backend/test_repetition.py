from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

recent_questions = []

def check_repetition(new_text, threshold=0.75):
    if not recent_questions:
        recent_questions.append(new_text)
        return False, None
    embeddings = model.encode([new_text] + recent_questions)
    sims = util.cos_sim(embeddings[0], embeddings[1:])
    max_sim = sims.max().item()
    recent_questions.append(new_text)
    return max_sim > threshold, max_sim

test_inputs = [
    "Where am I?",
    "What time is it?",
    "Where am I right now?",
    "Where is my mom?",
    "Where is this place?",
    "Where is the bathroom?",
]

for q in test_inputs:
    is_repeat, score = check_repetition(q)
    print(f"'{q}' -> repeat: {is_repeat}, similarity: {score}")