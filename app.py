import gc
import os

import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # reject uploads > 10 MB

MODEL_PATH = "landmark_resnet.pth"
DEVICE = torch.device("cpu")

# Keep CPU/RAM usage low on small instances (Render free tier = 512 MB)
torch.set_num_threads(1)

print("Using device:", DEVICE)

# ---------------------------------------------------------------------------
# Load checkpoint (memory-mapped, so the file isn't fully copied into RAM)
# Requires torch >= 2.1
# ---------------------------------------------------------------------------
print("Loading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,  # your own file; contains lists/strings
    mmap=True,
)

print("Checkpoint loaded.")

class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]
image_size = checkpoint["image_size"]
mean = checkpoint["mean"]
std = checkpoint["std"]
architecture = checkpoint.get("architecture", "ResNet50")

print("Architecture:", architecture)
print("Number of classes:", num_classes)
print("Image size:", image_size)
print("Classes:", class_names)

# ---------------------------------------------------------------------------
# Build model on the "meta" device (allocates no memory), then attach the
# memory-mapped weights directly with assign=True
# ---------------------------------------------------------------------------
with torch.device("meta"):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )

model.load_state_dict(checkpoint["model_state_dict"], assign=True)
model.eval()

del checkpoint
gc.collect()

print("ResNet50 model loaded successfully.")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Landmark Recognition API is running",
        "model": architecture,
        "classes": num_classes,
        "image_size": image_size,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image provided. Use form-data key: image",
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No image selected",
            }), 400

        image = Image.open(file).convert("RGB")
        image.thumbnail((1024, 1024))  # shrink big phone photos to save RAM

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)

        class_id = predicted_index.item()
        confidence_value = confidence.item()
        predicted_landmark = class_names[class_id]

        return jsonify({
            "success": True,
            "prediction": predicted_landmark,
            "class_id": class_id,
            "confidence": round(confidence_value * 100, 2),
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# ---------------------------------------------------------------------------
# Local run only (Render uses gunicorn)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )