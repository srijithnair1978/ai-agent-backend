from fastapi import FastAPI
import requests
import os
from dotenv import load_dotenv

# Load API keys from environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Agent Backend is Running!"}

# 📌 DeepSeek API Integration
@app.get("/deepseek/{query}")
def deepseek_query(query: str):
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": query}]
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# 📌 Google Search using SerpAPI
@app.get("/google/{query}")
def google_search(query: str):
    try:
        url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# 📌 Image Generation using Hugging Face
@app.get("/generate-image/{prompt}")
def generate_image(prompt: str):
    try:
        url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        response = requests.post(url, headers=headers, json={"inputs": prompt})
        
        if response.status_code == 200:
            return response.content
        return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}
