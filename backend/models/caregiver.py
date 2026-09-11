from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from bson import ObjectId

class CaregiverModel(BaseModel):
    """
    Conceptual model of the Caregiver document in MongoDB.
    """
    id: Optional[str] = None
    email: str
    password_hash: str
    name: str
    role: str = "caregiver"
    active: bool = True
    created_at: datetime
    updated_at: datetime

    def to_mongo(self) -> dict:
        doc = self.model_dump(exclude={"id"})
        if self.id:
            doc["_id"] = ObjectId(self.id)
        return doc

    @classmethod
    def from_mongo(cls, doc: dict) -> "CaregiverModel":
        if not doc:
            return None
        doc_copy = doc.copy()
        if "_id" in doc_copy:
            doc_copy["id"] = str(doc_copy.pop("_id"))
        return cls(**doc_copy)
