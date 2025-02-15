from fastapi import FastAPI, File, UploadFile
import wikipediaapi
import openai
import os
import fitz  # PyMuPDF for PDF analysis
import pandas as pd  # For Excel analysis
import pdfkit
import requests
from dotenv import load_dotenv
from io import BytesIO
from fastapi.responses import FileResponse
from diffusers import StableDiffusionPipeline
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# Get API keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

app = FastAPI()

# Enable CORS (Allow Frontend to Access Backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### **✅ Root Route (Fix for "GET / 404 Not Found")**
@app.get("/")
def home():
    return {"message": "AI Agent Backend is Running!"}


### **✅ Fix for "GET /favicon.ico 404 Not Found"**
@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon available"}


### **1️⃣ Wikipedia Search**
@app.get("/wikipedia/{query}")
def search_wikipedia(query: str):
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(query)
    return {"title": page.title, "summary": page.summary}


### **2️⃣ Google Search (Using SerpAPI)**
@app.get("/google/{query}")
def google_search(query: str):
    if not SERPAPI_KEY:
        return {"error": "Missing SERPAPI_KEY in .env"}
    
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    response = requests.get(url)
    return response.json()


### **3️⃣ OpenAI Query (Using GPT-3.5)**
@app.get("/openai/{query}")
def openai_query(query: str):
    if not OPENAI_API_KEY:
        return {"error": "Missing OPENAI_API_KEY in .env"}
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}],
        api_key=OPENAI_API_KEY
    )
    return {"response": response.choices[0].message["content"]}


### **4️⃣ PDF Upload & Analysis**
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    text = ""
    with fitz.open(stream=await file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return {"text": text}


### **5️⃣ Image Generation (Using Stable Diffusion)**
@app.get("/generate-image/{prompt}")
def generate_image(prompt: str):
    if not HUGGINGFACE_API_KEY:
        return {"error": "Missing HUGGINGFACE_API_KEY in .env"}
    
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    image = pipe(prompt).images[0]
    image_path = "generated_image.png"
    image.save(image_path)
    return FileResponse(image_path, media_type="image/png")


### **6️⃣ Data Analysis (Excel Upload & Custom Analysis)**
@app.post("/upload-excel/")
async def upload_excel(file: UploadFile = File(...)):
    df = pd.read_excel(BytesIO(await file.read()))
    return {"columns": list(df.columns), "data": df.to_dict()}


### **7️⃣ Flowchart Creation & PDF Export**
@app.post("/generate-flowchart/")
async def generate_flowchart(steps: str):
    flowchart_code = f"""
    graph TD;
    {steps.replace(",", " --> ")}
    """
    output_path = "flowchart.pdf"
    pdfkit.from_string(flowchart_code, output_path)
    return FileResponse(output_path, media_type='application/pdf')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
