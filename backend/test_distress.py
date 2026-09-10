DISTRESS_KEYWORDS = [
    "help", "scared", "afraid", "confused", "lost",
    "don't know where", "where am i", "who are you",
    "i want to go home", "hurts", "pain", "can't breathe"
]

def check_distress(text):
    text_lower = text.lower()
    matched = [kw for kw in DISTRESS_KEYWORDS if kw in text_lower]
    return len(matched) > 0, matched

test_inputs = [
    "I want to go home now",
    "What time is it?",
    "Help me, I'm scared",
    "Where is the bathroom?",
]

for text in test_inputs:
    is_distress, matched = check_distress(text)
    print(f"'{text}' -> distress: {is_distress}, matched: {matched}")