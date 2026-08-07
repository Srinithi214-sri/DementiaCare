from fastapi import FastAPI
from config.db import events_collection
from models.event import Event

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the DementiaCare OS API"}

@app.get("/events")
def get_events():
    events = list(events_collection.find())
    return events

@app.post("/events")
def create_event(event: Event):
    result = events_collection.insert_one(event.dict())
    return {"id": str(result.inserted_id)}