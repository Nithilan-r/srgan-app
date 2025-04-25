import os
import io
import torch
import uvicorn
import base64
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from model import GeneratorResNet
from utils import preprocess_image, postprocess_image, ensure_dir

# Constant downscale factor as requested
DOWNSCALE_FACTOR = 4

# Initialize FastAPI app
app = FastAPI(
    title="SRGAN Image Enhancer API",
    description="API for enhancing low-resolution images using SRGAN",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static files directory if it doesn't exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model loading function
def load_model():
    """Load the SRGAN generator model."""
    model = GeneratorResNet()
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model_path = os.path.join("saved_models", "generator.pth")
    if os.path.exists(model_path):
        # For Mac with M1/M2 chip
        if device.type == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model.load_state_dict(torch.load(model_path, map_location="mps"))
            model = model.to("mps")
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model.eval()
    return model, device

# Load model at startup
try:
    print("Loading SRGAN model...")
    model, device = load_model()
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    model, device = None, None

# Process image function
def process_image(image):
    """Process an image using the SRGAN model:
    1. Downscale by factor of 4 
    2. Process with model
    3. Return downscaled, SRGAN, and bicubic results
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Get original dimensions
    width, height = image.size
    
    # Downscale the image to simulate low resolution
    lr_width = width // DOWNSCALE_FACTOR
    lr_height = height // DOWNSCALE_FACTOR
    lr_image = image.resize((lr_width, lr_height), Image.BICUBIC)
    
    # Create bicubic upscale as baseline comparison
    bicubic_image = lr_image.resize((width, height), Image.BICUBIC)
    
    # Preprocess the low-resolution image for the model
    input_tensor = preprocess_image(lr_image)
    
    # Move tensor to appropriate device
    if device.type == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        input_tensor = input_tensor.to("mps")
    else:
        input_tensor = input_tensor.to(device)
    
    # Generate super-resolution image with the model
    with torch.no_grad():
        sr_tensor = model(input_tensor)
    
    # Convert back to PIL image
    sr_image = postprocess_image(sr_tensor)
    
    # Resize SR image to match original dimensions
    sr_image = sr_image.resize((width, height), Image.BICUBIC)
    
    return lr_image, bicubic_image, sr_image

# Helper function to convert PIL Image to base64
def image_to_base64(img, format="JPEG"):
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()

# Response model
class ImageResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/enhance", response_model=ImageResponse)
async def enhance_image(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and process image
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Process image with model
        lr_image, bicubic_image, sr_image = process_image(original_image)
        
        # Save images to disk
        timestamp = str(int(os.path.getmtime("static")))
        lr_path = f"static/results/lr_{timestamp}.jpg"
        bicubic_path = f"static/results/bicubic_{timestamp}.jpg"
        sr_path = f"static/results/sr_{timestamp}.jpg"
        
        lr_image.save(lr_path)
        bicubic_image.save(bicubic_path)
        sr_image.save(sr_path)
        
        # Convert images to base64 for response
        lr_b64 = image_to_base64(lr_image)
        bicubic_b64 = image_to_base64(bicubic_image)
        sr_b64 = image_to_base64(sr_image)
        
        return {
            "success": True,
            "message": "Image enhanced successfully",
            "data": {
                "lowres": {
                    "base64": f"data:image/jpeg;base64,{lr_b64}",
                    "path": f"/{lr_path}"
                },
                "bicubic": {
                    "base64": f"data:image/jpeg;base64,{bicubic_b64}",
                    "path": f"/{bicubic_path}"
                },
                "enhanced": {
                    "base64": f"data:image/jpeg;base64,{sr_b64}",
                    "path": f"/{sr_path}"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# API endpoint for test images
@app.get("/test-images", response_model=List[str])
async def get_test_images():
    test_dir = "Testset_20_LR"
    if not os.path.exists(test_dir):
        return []
    
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return test_images

@app.get("/test-image/{filename}", response_model=ImageResponse)
async def process_test_image(filename: str):
    test_dir = "Testset_20_LR"
    image_path = os.path.join(test_dir, filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Test image not found")
    
    try:
        # Read and process image
        original_image = Image.open(image_path).convert("RGB")
        
        # Process image with model
        lr_image, bicubic_image, sr_image = process_image(original_image)
        
        # Convert images to base64 for response
        lr_b64 = image_to_base64(lr_image)
        bicubic_b64 = image_to_base64(bicubic_image)
        sr_b64 = image_to_base64(sr_image)
        
        return {
            "success": True,
            "message": "Test image enhanced successfully",
            "data": {
                "lowres": {
                    "base64": f"data:image/jpeg;base64,{lr_b64}"
                },
                "bicubic": {
                    "base64": f"data:image/jpeg;base64,{bicubic_b64}"
                },
                "enhanced": {
                    "base64": f"data:image/jpeg;base64,{sr_b64}"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing test image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 