from fastapi import FastAPI
from src.models.art_generator import generate_art

app = FastAPI()


@app.post("/generate_art")
def create_art(prompt: str):
    image = generate_art(prompt)
    return {"image": image}
