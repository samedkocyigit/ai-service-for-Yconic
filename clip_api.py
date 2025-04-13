from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rembg import remove
from io import BytesIO
from typing import List
import clip
import torch
from PIL import Image, ImageDraw, ImageFont
import os
import math
import uuid

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
    outwear: List[ClotheItem] = []
    accessories: List[ClotheItem] = []

class SuggestionRequest(BaseModel):
    userPersona: str
    garderobe: Garderobe

def remove_background(img: Image.Image) -> Image.Image:
    try:
        with BytesIO() as buf:
            img.save(buf, format="PNG")
            result = remove(buf.getvalue())  # ✅ bytes veriyoruz
            return Image.open(BytesIO(result)).convert("RGBA")
    except Exception as e:
        print(f"Background removal failed: {e}")
        return img.convert("RGBA")


def create_combination_image(image_paths: list[str], save_dir="/app/wwwroot/uploads/combinations") -> str:
    grid_size = math.ceil(math.sqrt(len(image_paths)))
    cell_size = 256
    spacing = 10
    background_color = (30, 25, 40)

    canvas_width = (cell_size + spacing) * grid_size - spacing
    canvas_height = (cell_size + spacing) * grid_size - spacing
    canvas = Image.new("RGBA", (canvas_width, canvas_height), background_color)

    for idx, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGBA")
            img = remove_background(img) 
            img = img.resize((cell_size, cell_size))
            x = (idx % grid_size) * (cell_size + spacing)
            y = (idx // grid_size) * (cell_size + spacing)
            canvas.paste(img, (x, y), img)  
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            continue

    file_name = f"comb_{uuid.uuid4().hex[:8]}.png"
    full_path = os.path.join(save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)
    canvas.save(full_path)

    return f"/uploads/combinations/{file_name}"


# === Ana Endpoint: Kombin Önerisi ===
@app.post("/analyze-garderobe")
async def analyze_garderobe(payload: SuggestionRequest):
    base_uploads = "/app/wwwroot/uploads/clothe-photos"
    grouped_items = {
        "top": payload.garderobe.top,
        "bottom": payload.garderobe.bottom,
        "shoes": payload.garderobe.shoes,
        "outwear": payload.garderobe.outwear,
        "accessories": payload.garderobe.accessories
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

    # --- 🔥 Kolaj oluşturma adımı ---
    image_paths = [os.path.join(base_uploads, os.path.basename(item["image_path"])) for item in suggested_combination]
    main_image_url = create_combination_image(image_paths)

    return {
        "userPersona": payload.userPersona,
        "prompt": prompt,
        "suggested_combination": suggested_combination,
        "mainImageUrl": main_image_url
    }
