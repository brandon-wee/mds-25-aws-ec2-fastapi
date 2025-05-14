from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from typing import List

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from dotenv import load_dotenv
import os

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
# ▶️▶️▶️ NEW: load your recognition model + known embeddings at startup
# ——————————————————————————————————————————————————
MODEL_NAME = "buffalo_quantized_dynamic_v1"
EMB_CACHE_PATH = "embeddings_cache/buffalo_quantized_dynamic_v1_embeddings.pkl"
SIM_THRESHOLD = 0.3

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client.dashboard
users = db.users


def load_server_embeddings():
    """
    Fetch all documents with a non‐empty embeddings field,
    stack them into a (N × D) array, and return that array plus
    the corresponding list of usernames.
    """
    cursor = users.find(
        {"embeddings": {"$exists": True, "$ne": []}},
        {"username": 1, "embeddings": 1}
    )

    names = []
    embs = []
    for doc in cursor:
        emb_list = doc["embeddings"]
        # convert back to an ndarray
        embs.append(np.array(emb_list, dtype=np.float32))
        names.append(doc["username"])

    if embs:
        # (N, D) array
        known_embeddings = np.vstack(embs)
    else:
        # no known embeddings yet → empty array of shape (0, 0)
        known_embeddings = np.zeros((0, 0), dtype=np.float32)

    return known_embeddings, names


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


# Load FaceAnalysis once
face_app: FaceAnalysis = FaceAnalysis(name=MODEL_NAME)
face_app.prepare(ctx_id=0, det_size=(320, 320))

# Load known embeddings/names
known_embeddings, known_names = load_server_embeddings()

# In-memory store of last frame+meta
latest_frame: bytes | None = None
latest_meta: dict[str, any] = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Face Recognition Dashboard</title>
        <style>
          body { font-family: Arial, sans-serif; text-align: center; margin: 20px; }
          #video { border: 1px solid #ccc; }
          #metadata { margin-top: 20px; text-align: left; display: inline-block; }
          pre { background: #f4f4f4; padding: 10px; border-radius: 4px; }
        </style>
      </head>
      <body>
        <h1>Face Recognition Dashboard</h1>
        <p>Live Video Feed:</p>
        <img id="video" src="/video_feed" alt="Video Feed" width="640" />
        <div id="metadata">
          <h2>Latest Metadata</h2>
          <pre id="meta">Loading...</pre>
        </div>
        <script>
          async function fetchMetadata() {
            try {
              const res = await fetch('/metadata');
              const data = await res.json();
              document.getElementById('meta').textContent = JSON.stringify(data, null, 2);
            } catch (err) {
              document.getElementById('meta').textContent = 'Error fetching metadata';
            }
          }
          setInterval(fetchMetadata, 1000);
          fetchMetadata();
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


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


@app.post("/upload")
async def upload(
    frame: UploadFile = File(...),
    metadata: str = Form(...)
):
    global latest_frame, latest_meta

    # 1) Read in the frame and original metadata
    latest_frame = await frame.read()
    meta = json.loads(metadata)

    # 2) Decode JPEG → OpenCV image
    arr = np.frombuffer(latest_frame, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 3) For each bounding box, crop & recognize
    new_bboxes = []
    for box in meta.get("bboxes", []):
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        faces = face_app.get(crop)
        best_sim = 0
        if faces:
            emb = faces[0].normed_embedding.flatten()
            sims = np.dot(known_embeddings, emb)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            name = known_names[best_idx] if best_sim >= SIM_THRESHOLD else "Unknown"
        else:
            name = "NoFaceDetected"

        # Append identity to the box
        new_bboxes.append([x1, y1, x2, y2, name, best_sim])

    # 4) Overwrite metadata's boxes
    meta["bboxes"] = new_bboxes

    # Optional: log to file
    with open("logs.txt", "a") as logf:
        logf.write(json.dumps(meta) + "\n")

    latest_meta = meta
    return {"status": "recognized", "meta": meta}


@app.post("/calculate_average_embedding")
async def calculate_average_embedding(
    username: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Calculate an average face embedding for `username` from the uploaded images,
    update it in MongoDB, and return the new embedding.
    """
    # 1) Read & decode all uploaded images
    images = []
    for file in files:
        content = await file.read()
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        images.append(img)

    # 2) Compute average embedding
    avg_embedding = calculate_average_face_embedding(images)
    if avg_embedding is None:
        return {"status": "error", "message": "No faces detected in any of the images"}

    # 3) Persist it
    success = update_embedding(username, avg_embedding)
    if not success:
        return {"status": "error", "message": f"User '{username}' not found in database"}

    # 4) Return JSON‐serializable list
    return {
        "status":   "success",
        "username": username,
        "embedding": avg_embedding.tolist(),
        "count":    len(images)
    }


async def mjpeg_generator():
    global latest_frame, latest_meta
    while True:
        if latest_frame:
            # 1) decode
            arr = np.frombuffer(latest_frame, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            # 2) draw boxes + names
            for box in latest_meta.get("bboxes", []):
                x1, y1, x2, y2, name, sim = box
                # box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # name
                cv2.putText(
                    img, name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

            # 3) re-encode
            _, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()

            # 4) yield
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )

        await asyncio.sleep(0.05)


@app.get("/video_feed")
def video_feed():
    """
    Clients can point an <img> tag or fetch this URL
    and get a multipart/x-mixed-replace MJPEG stream.
    """
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/metadata")
def metadata():
    """
    Returns the latest metadata as JSON:
      { "fps": ..., "people_count": ..., ... }
    """
    return JSONResponse(content=latest_meta)
