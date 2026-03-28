import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import io
from pathlib import Path
from model import Net

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mnist_model.pth"

app = FastAPI()
model = Net()

# Load the weights created by train.py during the Docker build.
# Using an absolute path avoids cwd-related failures.
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model weights not found at {MODEL_PATH}. Run train.py (or Docker build) first."
    )
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

model.eval()

# Image preprocessing
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    tensor = transform_image(data)
    with torch.no_grad():
        outputs = model(tensor)
    _, y_hat = outputs.max(1)
    return {"prediction": int(y_hat.item())}

@app.get("/")
def read_root():
    return {"message": "MNIST Prediction API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}