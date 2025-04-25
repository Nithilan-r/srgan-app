# SRGAN Image Enhancer

A Streamlit application that uses Super-Resolution Generative Adversarial Networks (SRGAN) to enhance low-resolution images.

## Features

- Upload your own images for enhancement
- Use sample test images
- Adjust downscaling factor
- Compare original, low-resolution, and enhanced images side by side
- Download enhanced images

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `model.py`: SRGAN model architecture
- `utils.py`: Utility functions for image processing
- `saved_models/`: Directory containing pretrained model weights
- `Testset_20_LR/`: Directory containing test images

## Model Information

The SRGAN model used in this application is a deep learning model that can generate super-resolution (SR) images from low-resolution inputs. The model was trained to upscale images by 4x while preserving details and enhancing clarity.

## Requirements

- Python 3.8+
- PyTorch
- CUDA-compatible GPU (optional, for faster processing)

## Notes for Mac Users

The application includes special handling for Mac with M1/M2 chips using MPS (Metal Performance Shaders) acceleration.

## License

This project is licensed under the MIT License. 