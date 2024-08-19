import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


def get_clip_model():
    model_path = os.getenv("EMBEDDING_MODEL_PATH")
    return CLIPModel.from_pretrained(model_path)


def get_clip_processor():
    processor_path = os.getenv("EMBEDDING_PROCESSOR_PATH")
    return CLIPProcessor.from_pretrained(processor_path)


def generate_clip_embedding_from_image(image_path, model, processor):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs).squeeze().tolist()
        return embeddings
    except Exception as e:
        print(f"Error generating embedding for {image_path}: {e}")
        return None


def generate_clip_embedding_from_text(text, model, processor):
    inputs = processor(text=text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features[0].cpu().numpy()
