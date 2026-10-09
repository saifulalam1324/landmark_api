from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


app = Flask(__name__)


MODEL_PATH = "landmark_resnet.pth"
DEVICE = torch.device("cpu")

print("Using device:", DEVICE)

print("Loading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint loaded.")


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


model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(
        model.fc.in_features,
        num_classes
    )
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

# Disable gradient calculation for the entire model
for parameter in model.parameters():
    parameter.requires_grad = False

print("ResNet50 model loaded successfully.")


del checkpoint

print("Checkpoint removed from memory.")



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


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Landmark Recognition API is running",
        "model": architecture,
        "classes": num_classes,
        "image_size": image_size
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:


        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": (
                    "No image provided. "
                    "Use form-data key: image"
                )
            }), 400

        file = request.files["image"]


        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No image selected"
            }), 400


        image = Image.open(file).convert("RGB")


        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(DEVICE)



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



        class_id = predicted_index.item()

        confidence_value = confidence.item()

        predicted_landmark = class_names[class_id]



        del image_tensor
        del outputs
        del probabilities



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



if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )