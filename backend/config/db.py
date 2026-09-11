import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from config.settings import settings

class Database:
    client: MongoClient = None
    db = None

    @classmethod
    def connect(cls):
        if cls.client is not None:
            return
        try:
            cls.client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2000)
            cls.client.admin.command('ping')
            cls.db = cls.client["dementiacare"]
            logging.info("Connected to MongoDB successfully.")
            
            # Idempotently create indexes
            try:
                cls.db["caregivers"].create_index([("email", 1)], unique=True)
                cls.db["patients"].create_index([("active", 1)])
                cls.db["patients"].create_index([("created_at", -1)])
                cls.db["events"].create_index([("patient_id", 1)])
                cls.db["events"].create_index([("timestamp", -1)])
                cls.db["events"].create_index([("patient_id", 1), ("source", 1), ("timestamp", -1)])
                cls.db["memories"].create_index([("patient_id", 1), ("active", 1), ("approved", 1)])
                cls.db["memories"].create_index([("patient_id", 1), ("created_at", -1)])
            except Exception as e:
                logging.warning(f"Failed to create indexes during startup: {e}")
                
        except PyMongoError:
            logging.error("Failed to connect to MongoDB. Verify configuration.")
            raise

    @classmethod
    def close(cls):
        if cls.client:
            cls.client.close()
            logging.info("MongoDB connection closed.")

    @classmethod
    def get_events_collection(cls):
        if cls.db is None:
            raise RuntimeError("Database not initialized. Please call connect() first.")
        return cls.db["events"]

    @classmethod
    def get_caregivers_collection(cls):
        if cls.db is None:
            raise RuntimeError("Database not initialized. Please call connect() first.")
        return cls.db["caregivers"]

    @classmethod
    def get_patients_collection(cls):
        if cls.db is None:
            raise RuntimeError("Database not initialized. Please call connect() first.")
        return cls.db["patients"]

    @classmethod
    def get_memories_collection(cls):
        if cls.db is None:
            raise RuntimeError("Database not initialized. Please call connect() first.")
        return cls.db["memories"]

db = Database()