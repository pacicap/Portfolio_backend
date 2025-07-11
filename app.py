from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline
import torch
import uuid
from PIL import Image
import os
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "https://react-portfolio-git-main-pacicaps-projects.vercel.app",
                   "https://react-portfolio-ij8ifou62-pacicaps-projects.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Remote Hugging Face model IDs
hf_model_ids = {
    "model1": "Pacicap/FineTuned_claude_StableDiffussion_2_1",
    "model2": "Pacicap/FineTuned_gpt4o_StableDiffussion_2_1"
}

loaded_models = {}

class PromptInput(BaseModel):
    prompt: str
    model: str  # should be "model1" or "model2"

@app.post("/generate")
def generate(data: PromptInput, request: Request):
    model_key = data.model

    if model_key not in hf_model_ids:
        return {"error": "Invalid model selected"}

    model_id = hf_model_ids[model_key]

    if model_key not in loaded_models:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = DiffusionPipeline.from_pretrained(model_id).to(device)
        loaded_models[model_key] = pipe
    else:
        pipe = loaded_models[model_key]

    image = pipe(data.prompt).images[0]

    os.makedirs("generated", exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join("generated", filename)
    image.save(filepath)

    image_url = f"{request.base_url}generated/{filename}"
    return {"url": image_url}

    #return {"url": f"http://localhost:8000/generated/{filename}"}

# Serve images
app.mount("/generated", StaticFiles(directory="generated"), name="generated")
