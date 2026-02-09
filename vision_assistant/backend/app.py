from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64, uuid, os

from vision import query_vlm, LIVE_SYSTEM_PROMPT
from video import summarize_video
from memory import is_new_scene

app = FastAPI()

app.mount("/frontend", StaticFiles(directory="../frontend"), name="frontend")

@app.get("/")
def home():
    return FileResponse("../frontend/index.html")

@app.get("/live")
def live():
    return FileResponse("../frontend/live.html")

# ---------- MODELS ----------
class LiveFrameRequest(BaseModel):
    image: str

class LiveQuestionRequest(BaseModel):
    image: str
    question: str

# ---------- IMAGE ----------
@app.post("/image")
async def image_summary(file: UploadFile):
    img = await file.read()
    b64 = "data:image/jpeg;base64," + base64.b64encode(img).decode()

    summary = query_vlm(
        b64,
        (
            "You are summarizing an image.\n"
            "Describe all clearly visible people, objects, and actions.\n"
            "Do not guess or infer.\n"
            "Write a complete but concise paragraph."
        ),
        profile="image_summary"
    )
    return {"summary": summary}

# ---------- VIDEO ----------
@app.post("/video")
async def video_summary(file: UploadFile):
    name = f"tmp_{uuid.uuid4()}.mp4"
    with open(name, "wb") as f:
        f.write(await file.read())

    summary = summarize_video(name)
    os.remove(name)

    return {"summary": summary}

# ---------- LIVE FRAME ----------
@app.post("/live_frame")
async def live_frame(req: LiveFrameRequest):
    desc = query_vlm(
        req.image,
        LIVE_SYSTEM_PROMPT,
        profile="live"
    )

    if is_new_scene(desc):
        return {"scene": desc}

    return {"scene": ""}

# ---------- LIVE QUESTION ----------
@app.post("/live_question")
async def live_question(req: LiveQuestionRequest):
    answer = query_vlm(
        req.image,
        (
            "You are answering a question about a live camera scene.\n"
            "Rules:\n"
            "- Answer ONLY the question.\n"
            "- Do NOT describe the scene.\n"
            "- Do NOT add extra context.\n"
            "- If the answer cannot be determined from the image, reply exactly:\n"
            "  'Cannot determine from the scene.'\n"
            "- Answer in ONE short sentence."
        ),
        req.question,
        profile="qa"
    )
    return {"answer": answer}
