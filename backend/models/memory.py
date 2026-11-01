"""
Conceptual representation of the Memory document in MongoDB.
Pydantic schemas in schemas/memory.py are used for API validation and serialization.

MongoDB Document Structure:
{
    "_id": ObjectId,
    "patient_id": str (stored as string for simple isolation, or ObjectId),
    "title": str,
    "content": str,
    "category": str (Optional),
    "date_reference": str (Optional),
    "people": list[str] (Optional),
    "location": str (Optional),
    "source": str,
    "approved": bool,
    "active": bool,
    "created_at": datetime,
    "updated_at": datetime
}
"""
