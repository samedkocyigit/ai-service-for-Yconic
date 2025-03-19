from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import clip
import torch
from PIL import Image
import os

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

@app.get("/health")
def health_check():
    return {"status": "ok"}

class ClotheItem(BaseModel):
    clotheId: str
    image_path: str

class Garderobe(BaseModel):
    top: List[ClotheItem] = []
    bottom: List[ClotheItem] = []
    shoes: List[ClotheItem] = []
    outerwear: List[ClotheItem] = []
    accessory: List[ClotheItem] = []

class SuggestionRequest(BaseModel):
    userPersona: str
    garderobe: Garderobe

@app.post("/analyze-garderobe")
async def analyze_garderobe(payload: SuggestionRequest):
    base_uploads = "/app/wwwroot/uploads"
    grouped_items = {
        "top": payload.garderobe.top,
        "bottom": payload.garderobe.bottom,
        "shoes": payload.garderobe.shoes,
        "outerwear": payload.garderobe.outerwear,
        "accessory": payload.garderobe.accessory
    }

    suggested_combination = []

    prompt = (
        f"You are a high-end fashion stylist. Based on the '{payload.userPersona}' style and today's weather being 'cold', "
        "choose the best clothing combination from the items provided. Select only the essential pieces for today’s conditions, ensuring style, comfort, and practicality."
    )

    for group_name, clothes in grouped_items.items():
        if not clothes:
            continue

        images_tensor = []
        valid_items = []

        for item in clothes:
            full_path = os.path.join(base_uploads, os.path.basename(item.image_path))
            if os.path.exists(full_path):
                image = Image.open(full_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)
                images_tensor.append(image_input)
                valid_items.append(item)

        if not images_tensor:
            continue

        batch_images = torch.cat(images_tensor)
        text_tokens = clip.tokenize([prompt]).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(batch_images, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        best_idx = probs[:, 0].argmax()
        best_item = valid_items[best_idx]
        suggested_combination.append({
            "clotheId": best_item.clotheId,
            "image_path": best_item.image_path
        })

    # if payload.weather == "warm":
    #     outerwear_ids = [item.clotheId for item in payload.garderobe.outerwear]
    #     suggested_combination = [
    #         item for item in suggested_combination if item["clotheId"] not in outerwear_ids
    #     ]

    return {
        "userPersona": payload.userPersona,
        "prompt": prompt,
        "suggested_combination": suggested_combination
    }
