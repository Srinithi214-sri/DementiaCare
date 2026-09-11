import logging

def normalize_memory_text(memory: dict) -> str:
    """
    Extracts and normalizes safe textual fields from a memory dictionary
    to prepare for semantic embedding.
    
    CRITICAL: Never include IDs, credentials, or raw media.
    """
    if not memory:
        return ""
        
    parts = []
    
    title = memory.get("title", "")
    if title:
        parts.append(f"Title: {title.strip()}")
        
    content = memory.get("content", "")
    if content:
        parts.append(f"Content: {content.strip()}")
        
    category = memory.get("category", "")
    if category:
        parts.append(f"Category: {category.strip()}")
        
    date_ref = memory.get("date_reference", "")
    if date_ref:
        parts.append(f"Date: {date_ref.strip()}")
        
    # Join parts
    normalized_text = "\n".join(parts)
    
    # Bound the length to a safe maximum (e.g. 8000 characters) to prevent OOM
    MAX_TEXT_LEN = 8000
    if len(normalized_text) > MAX_TEXT_LEN:
        logging.warning(f"Memory text exceeded max length {MAX_TEXT_LEN}, truncating before embedding.")
        normalized_text = normalized_text[:MAX_TEXT_LEN]
        
    return normalized_text
