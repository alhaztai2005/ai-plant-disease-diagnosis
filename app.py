from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.resnet50 import preprocess_input

import io
import logging
from pathlib import Path

# ======================
# LOGGING
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# PATH CONFIG
# ======================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "bestModel.keras"

# ======================
# FASTAPI APP
# ======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# STATIC & TEMPLATES
# ======================
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ======================
# LOAD MODEL
# ======================
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading error: {e}")
    raise RuntimeError("Failed to load the model.")

# ======================
# CLASS NAMES
# ======================
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ======================
# DISEASE TREATMENT DATA
# ======================
disease_treatment = {
    "Tomato___Bacterial_spot": {
        "description": "A bacterial disease causing dark, wet spots on leaves and fruits.",
        "treatment": [
            "Apply copper-based bactericides.",
            "Remove and destroy infected plants.",
            "Avoid overhead watering."
        ]
    },
    "Tomato___Early_blight": {
        "description": "A fungal disease causing dark spots with concentric rings on leaves.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Remove infected leaves and plants.",
            "Rotate crops to prevent soil-borne fungi."
        ]
    },
    "Tomato___Late_blight": {
        "description": "A fungal disease causing dark, water-soaked lesions on leaves and stems.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Remove and destroy infected plants.",
            "Avoid overhead watering and ensure proper spacing for air circulation."
        ]
    },
    "Tomato___Leaf_Mold": {
        "description": "A fungal disease causing yellow spots on the upper leaf surface and mold on the underside.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Improve air circulation by spacing plants properly.",
            "Avoid overhead watering."
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "A fungal disease causing small, circular spots with gray centers and dark edges on leaves.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Remove infected leaves and destroy them.",
            "Avoid overhead watering."
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "A pest infestation causing yellow stippling on leaves and fine webbing on the plant.",
        "treatment": [
            "Apply insecticidal soap or neem oil.",
            "Increase humidity to deter mites.",
            "Remove heavily infested leaves."
        ]
    },
    "Tomato___Target_Spot": {
        "description": "A fungal disease causing dark, target-like spots on leaves.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Remove infected leaves and destroy them.",
            "Avoid overhead watering."
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "A viral disease causing yellowing and curling of leaves, stunted growth, and reduced fruit production.",
        "treatment": [
            "Remove and destroy infected plants.",
            "Control whitefly populations (the virus vector) using insecticides.",
            "Plant resistant varieties if available."
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "A viral disease causing mottled leaves, stunted growth, and reduced fruit yield.",
        "treatment": [
            "Remove and destroy infected plants.",
            "Disinfect tools and hands to prevent spread.",
            "Plant resistant varieties if available."
        ]
    },
    "Tomato___healthy": {
        "description": "The plant is healthy and shows no signs of disease.",
        "treatment": [
            "Continue good cultural practices.",
            "Monitor for early signs of disease.",
            "Maintain proper watering and fertilization."
        ]
    }
}

# ======================
# IMAGE PREPROCESSING
# ======================
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255.0  # Important for ResNet
    image = np.expand_dims(image, axis=0)
    return image

# ======================
# ROUTES
# ======================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        image = Image.open(io.BytesIO(contents))

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        predicted_class = class_names[predicted_index]

        treatment_info = disease_treatment.get(predicted_class, {
            "description": "No description available.",
            "treatment": ["No treatment available."]
        })

        return JSONResponse({
            "predicted_disease": predicted_class,
            "confidence": round(confidence * 100, 2),
            "description": treatment_info["description"],
            "treatment": treatment_info["treatment"]
        })

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# RUN SERVER
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
