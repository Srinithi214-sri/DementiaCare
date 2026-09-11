import os
import sys
from datetime import datetime, timezone
import logging

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.db import db
from auth.password import hash_password

def create_admin():
    db.connect()
    collection = db.get_caregivers_collection()
    
    admin_email = os.environ.get("ADMIN_EMAIL")
    admin_password = os.environ.get("ADMIN_PASSWORD")
    admin_name = os.environ.get("ADMIN_NAME", "System Administrator")
    
    if not admin_email or not admin_password:
        logging.error("ADMIN_EMAIL and ADMIN_PASSWORD environment variables are required.")
        sys.exit(1)
        
    admin_email = admin_email.lower().strip()
    
    if collection.find_one({"email": admin_email}):
        logging.info(f"Admin account {admin_email} already exists.")
        sys.exit(0)
        
    now = datetime.now(timezone.utc)
    hashed_password = hash_password(admin_password)
    
    admin_doc = {
        "email": admin_email,
        "password_hash": hashed_password,
        "name": admin_name,
        "role": "admin",
        "active": True,
        "created_at": now,
        "updated_at": now
    }
    
    collection.insert_one(admin_doc)
    logging.info(f"Successfully created initial admin account for {admin_email}.")
    db.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_admin()
