from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import clip
import torch
from PIL import Image
import os

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define Pydantic models for the request
class ClotheItem(BaseModel):
    clotheId: str  # Use string; conversion to UUID can be done later if needed
    image_path: str

class Garderobe(BaseModel):
    # Dynamic categories; keys are category names and values are lists of ClotheItem
    categories: Dict[str, List[ClotheItem]]

class SuggestionRequest(BaseModel):
    userPersona: str
    garderobe: Garderobe

@app.post("/analyze-garderobe")
async def analyze_garderobe(payload: SuggestionRequest):
    # Flatten the clothing items from all dynamic categories
    items = []
    for category_items in payload.garderobe.categories.values():
        items.extend(category_items)

    if not items:
        raise HTTPException(status_code=400, detail="No clothing items provided")

    images_tensor = []
    for item in items:
        # Adjust the base path as needed; this should match your Docker volume mount
        base_uploads = "/app/wwwroot/uploads"
        # Use os.path.basename to ensure we only use the filename part
        full_path = os.path.join(base_uploads, os.path.basename(item.image_path))
        if not os.path.exists(full_path):
            raise HTTPException(status_code=400, detail=f"Image not found: {full_path}")
        image = Image.open(full_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        images_tensor.append(image_input)
    
    # Create a batch of images from the list of tensors
    batch_images = torch.cat(images_tensor)

    # Build a dynamic prompt using the user's persona
    prompt = (
        f"As a high-end fashion designer, analyze these clothing pieces for a '{payload.userPersona}' style. "
        "Suggest the most stylish combination by selecting the best 2 or 3 items that work well together in terms of color, fabric, and trend."
    )
    
    # Define candidate text prompts for comparison
    text_prompts = [prompt, "minimalist outfit", "luxury outfit", "casual outfit"]
    text_tokens = clip.tokenize(text_prompts).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(batch_images, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Build the result list, including each clotheId and its scores
    results = []
    for i, item in enumerate(items):
        result = {
            "clotheId": item.clotheId,
            "image_path": item.image_path,
            "primary_score": float(probs[i][0]),
            "minimalist_score": float(probs[i][1]),
            "luxury_score": float(probs[i][2]),
            "casual_score": float(probs[i][3])
        }
        results.append(result)

    # Sort by primary score (you can adjust this logic as needed)
    sorted_results = sorted(results, key=lambda x: x["primary_score"], reverse=True)
    suggested_combination = sorted_results[:3]

    return {
        "userPersona": payload.userPersona,
        "prompt": prompt,
        "suggested_combination": suggested_combination
    }
