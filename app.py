from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os


# ============================================================
# Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "landmark_resnet.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# ============================================================
# Load checkpoint
# ============================================================

print("Loading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

print("Checkpoint loaded.")


# ============================================================
# Read information stored inside checkpoint
# ============================================================

class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]
image_size = checkpoint["image_size"]

mean = checkpoint["mean"]
std = checkpoint["std"]

architecture = checkpoint.get("architecture", "Unknown")

print("Architecture:", architecture)
print("Number of classes:", num_classes)
print("Image size:", image_size)
print("Classes:", class_names)


# ============================================================
# Create ResNet50
# ============================================================

model = models.resnet50(weights=None)


# The checkpoint contains:
#
# fc.1.weight
# fc.1.bias
#
# This means the original FC layer was replaced by
# a Sequential layer.
#
# fc.0 = Dropout
# fc.1 = Linear
#
# Dropout probability does not affect prediction because
# model.eval() is used during inference.

model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, num_classes)
)


# ============================================================
# Load trained weights
# ============================================================

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)

model.eval()

print("ResNet50 model loaded successfully.")


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )
])


# ============================================================
# Health check
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Landmark Recognition API is running",
        "model": architecture,
        "classes": num_classes,
        "image_size": image_size
    })


# ============================================================
# Prediction endpoint
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Check whether image exists
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image provided. Use form-data key: image"
            }), 400


        file = request.files["image"]


        # ----------------------------------------------------
        # Check filename
        # ----------------------------------------------------

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No image selected"
            }), 400


        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(file).convert("RGB")


        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        image_tensor = transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(DEVICE)


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )


        # ----------------------------------------------------
        # Convert result
        # ----------------------------------------------------

        class_id = predicted_index.item()

        confidence_value = confidence.item()

        predicted_landmark = class_names[class_id]


        # ----------------------------------------------------
        # Return JSON
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "prediction": predicted_landmark,

            "class_id": class_id,

            "confidence": round(
                confidence_value * 100,
                2
            )

        })


    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# Run server
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )