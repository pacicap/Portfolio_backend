from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline
import torch
import uuid
import os
from PIL import Image
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accept from all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hf_model_ids = {
    "model1": "Pacicap/FineTuned_claude_StableDiffussion_2_1",
    "model2": "Pacicap/FineTuned_gpt4o_StableDiffussion_2_1"
}

loaded_models = {}

class PromptInput(BaseModel):
    prompt: str
    model: str

@app.post("/generate")
def generate(data: PromptInput, request: Request):
    model_key = data.model

    if model_key not in hf_model_ids:
        return {"error": "Invalid model selected"}

    model_id = hf_model_ids[model_key]

    if model_key not in loaded_models:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        ).to("cpu")  # CPU-safe for Spaces
        loaded_models[model_key] = pipe
    else:
        pipe = loaded_models[model_key]

    image = pipe(data.prompt).images[0]

    os.makedirs("generated", exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join("generated", filename)
    image.save(filepath)

    return {
        "url": f"{request.base_url}generated/{filename}"
    }

app.mount("/generated", StaticFiles(directory="generated"), name="generated")
