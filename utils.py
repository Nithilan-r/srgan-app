import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import os

def load_image(image_path):
    """Load an image and convert to RGB."""
    img = Image.open(image_path).convert('RGB')
    return img

def preprocess_image(image, target_size=None):
    """Preprocess an image for the SRGAN model."""
    if target_size:
        image = image.resize((target_size, target_size), Image.BICUBIC)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    return transform(image).unsqueeze(0)

def postprocess_image(tensor):
    """Convert output tensor to PIL Image."""
    # Denormalize
    tensor = tensor.clone().detach()
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL Image
    tensor = tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    
    return Image.fromarray(tensor)

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory) 