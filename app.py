from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================================
# Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "landmark_resnet.pth"

# Render Free uses CPU.
# Force CPU instead of checking for CUDA.
DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ============================================================
# Load checkpoint
# ============================================================

print("Loading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint loaded.")


# ============================================================
# Read checkpoint information
# ============================================================

class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]
image_size = checkpoint["image_size"]

mean = checkpoint["mean"]
std = checkpoint["std"]

architecture = checkpoint.get(
    "architecture",
    "ResNet50"
)

print("Architecture:", architecture)
print("Number of classes:", num_classes)
print("Image size:", image_size)
print("Classes:", class_names)


# ============================================================
# Create ResNet50
# ============================================================

model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(
        model.fc.in_features,
        num_classes
    )
)


# ============================================================
# Load trained weights
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

# Disable gradient calculation for the entire model
for parameter in model.parameters():
    parameter.requires_grad = False

print("ResNet50 model loaded successfully.")


# ============================================================
# Delete checkpoint from memory
# ============================================================

del checkpoint

print("Checkpoint removed from memory.")


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (image_size, image_size)
    ),

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
        # Check image
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": (
                    "No image provided. "
                    "Use form-data key: image"
                )
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

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(DEVICE)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        class_id = predicted_index.item()

        confidence_value = confidence.item()

        predicted_landmark = class_names[class_id]

        # ----------------------------------------------------
        # Release image tensors
        # ----------------------------------------------------

        del image_tensor
        del outputs
        del probabilities

        # ----------------------------------------------------
        # Return response
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

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# Local development
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )