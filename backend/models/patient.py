"""
Conceptual representation of the Patient document in MongoDB.
Pydantic schemas in schemas/patient.py are used for API validation and serialization.

MongoDB Document Structure:
{
    "_id": ObjectId,
    "name": str,
    "preferred_name": str,
    "language": str,
    "timezone": str,
    "active": bool,
    "created_at": datetime,
    "updated_at": datetime
}
"""
