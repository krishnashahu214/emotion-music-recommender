from fastapi import FastAPI, File, UploadFile, Response, status, HTTPException, Depends, Form
import tempfile
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from models.face_emotion import get_face_emotion
from models.text_emotion import get_text_emotion
from models.fusion import fuse_emotions
from recommender.music_mapper import emotion_to_genre
from utils.spotify_api import get_songs_by_genre
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"] #the website which can access

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_TYPES = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/webp",
    "image/tiff"
]

async def validate_file(file: UploadFile):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only video files are allowed"
        )
    return file

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "user_images")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/analyze")
async def analyze(text: str= Form(...), file : UploadFile = Depends(validate_file)):

    # Validate filename exists
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a valid filename"
        )
    
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Save file to your custom folder
    with open(image_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    face_emotion, f_confidence = get_face_emotion(image_path)
    text_emotion, t_confidence = get_text_emotion(text)

    t_confidence = t_confidence * 100

    final_emotion = fuse_emotions(f_confidence, t_confidence, face_emotion, text_emotion)

    genres = emotion_to_genre.get(final_emotion, ["chill"])
    songs = get_songs_by_genre(genres)

    return {
        "face_emotion": face_emotion,
        "face_confidence": f_confidence,
        "text_emotion": text_emotion,
        "text_confidence": t_confidence,
        "final_emotion": final_emotion,
        "songs": songs
    }