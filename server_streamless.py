from fastapi import FastAPI, Body, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import cv2
import numpy as np
import base64
from insightface.app import FaceAnalysis
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from typing import Dict, List

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ——————————————————————————————————————————————————
# ▶️ Load face model + known embeddings at startup
# ——————————————————————————————————————————————————
MODEL_NAME = "buffalo_sc"
SIM_THRESHOLD = 0.3

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = client.dashboard
users = db.users
photo = 0


def load_server_embeddings():
    cursor = users.find(
        {"embeddings": {"$exists": True, "$ne": []}},
        {"username": 1, "embeddings": 1}
    )
    names, embs = [], []
    for doc in cursor:
        embs.append(np.array(doc["embeddings"], dtype=np.float32))
        names.append(doc["username"])
    known_embeddings = np.vstack(
        embs) if embs else np.zeros((0, 0), dtype=np.float32)
    return known_embeddings, names


known_embeddings, known_names = load_server_embeddings()
print(known_names)
face_app = FaceAnalysis(name=MODEL_NAME)
face_app.prepare(ctx_id=0, det_size=(320, 320))

latest_meta: Dict = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(
        "<h1>Face Recognition API</h1>"
        "<p>POST your metadata JSON to <code>/recognize</code></p>"
    )


@app.post("/recognize")
async def recognize(metadata: Dict = Body(...)):
    """
    Expects JSON like:
    {
      "bboxes": [
        {
          "bbox": [x1, y1, x2, y2],
          "crop": "<base64-jpg-string>"
        },
        ...
      ],
      ... any other fields ...
    }
    """
    global latest_meta, photo
    meta = metadata
    unknown = 0

    enriched = []
    for entry in meta.get("bboxes", []):
        # preserve bbox + crop
        bbox = entry["bbox"]
        crop_b64 = entry["crop"]

        # decode the base64 crop to CV image
        img_data = base64.b64decode(crop_b64)

        arr = np.frombuffer(img_data, np.uint8)
        crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # run insightface
        faces = face_app.get(crop)
        if faces:
            emb = faces[0].normed_embedding.flatten()
            sims = np.dot(known_embeddings, emb)
            idx = int(np.argmax(sims))
            sim_val = float(sims[idx])
            if sim_val >= SIM_THRESHOLD:
                name = known_names[idx]
            else:
                name = f"Unknown {unknown}"
                unknown += 1
            with open(f"photo/photo_{name}_{photo}.png", "wb+") as file:
                file.write(img_data)
                photo += 1
        else:
            name, sim_val = "NoFaceDetected", 0.0

        # build the enriched entry
        enriched.append({
            "bbox":       bbox,
            "crop":       crop_b64,
            "name":       name,
            "similarity": sim_val
        })

    # overwrite with enriched list
    meta["bboxes"] = enriched
    latest_meta = meta

    # append to log
    with open("logs.txt", "a") as logf:
        logf.write(json.dumps(meta) + "\n")

    return {"status": "recognized", "meta": meta}


@app.get("/metadata")
async def get_latest_metadata():
    return JSONResponse(content=latest_meta)


# FIX DOWN HERE! Add user embeddings to mongodb database

def update_embedding(username, embedding):
    try:
        embedding = embedding.tolist()
    except AttributeError:
        # already a list
        embedding = embedding
    # 1) Try to fetch the existing embeddings
    user = users.find_one({"username": username}, {"embeddings": 1})

    if user:
        old_embeddings = user.get("embeddings", [])
        # If lengths differ you might decide to pad or skip averaging
        if len(old_embeddings) == len(embedding):
            # 2) Compute element-wise average
            averaged = [
                (old + new) / 2.0
                for old, new in zip(old_embeddings, embedding)
            ]
        else:
            # fallback: just overwrite if lengths don't match
            averaged = embedding

        # 3) Write the averaged array back
        result = users.update_one(
            {"username": username},
            {"$set": {"embeddings": averaged}}
        )
        print(
            f"Matched {result.matched_count}, modified {result.modified_count}")
        return True
    else:
        # No such user—insert a new document (or handle as you prefer)
        print("User not found!")
        return False


def calculate_average_face_embedding(images: List[np.ndarray]) -> np.ndarray:
    """
    Calculate the average face embedding from a list of images.

    Args:
        images: List of numpy arrays containing images

    Returns:
        np.ndarray: The average face embedding vector or None if no faces found
    """
    if not images or len(images) == 0:
        return None

    all_embeddings = []

    # Process each image and extract face embeddings
    for img in images:
        faces = face_app.get(img)
        if faces:
            # Get the embedding from the first detected face
            emb = faces[0].normed_embedding.flatten()
            all_embeddings.append(emb)

    # If no faces were detected in any image, return None
    if not all_embeddings:
        return None

    # Calculate the average embedding
    avg_embedding = np.mean(all_embeddings, axis=0)

    # Normalize the average embedding
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm

    return avg_embedding


@app.post("/calculate_average_embedding")
async def calculate_average_embedding(
    username: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Upload multiple images for `username`, compute their average face
    embedding, update MongoDB, and return the new embedding.
    """
    global known_embeddings, known_names
    images = []
    for file in files:
        content = await file.read()
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    avg_emb = calculate_average_face_embedding(images)
    if avg_emb is None:
        return {"status": "error", "message": "No faces detected in any image."}

    success = update_embedding(username, avg_emb)
    if not success:
        return {"status": "error", "message": f"User '{username}' not found."}

    known_embeddings, known_names = load_server_embeddings()
    print(known_names)
    return {
        "status":    "success",
        "username":  username,
        "embedding": avg_emb.tolist(),
        "images":    len(images)
    }
