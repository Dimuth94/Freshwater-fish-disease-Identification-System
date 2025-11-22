import argparse
import torch, cv2
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from pathlib import Path
from src.gradcam import gradcam_on_image, overlay_cam

# --- CLI args ---
parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default=None, help="Path to a test image")
args = parser.parse_args()

CKPT = Path("outputs/best.pt")
#IMG  = r"D:\MSC\Research\Freshwater-fish-disease-Identification-System\freshwater_fish_disease\data\dataset\test\bacterial_aeromoniasis\bacterial_diseases_aeromoniasis_008.jpg"
IMG_SIZE = 224

# --- load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT, map_location="cpu")
class_names = ckpt["classes"]

#effiecinet50 code part
# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# in_feat = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(in_feat, len(class_names))
# model.load_state_dict(ckpt["model"])
# model.to(device).eval()

# Build ResNet50 model 
model = models.resnet50(weights=None)
# Replace final fully connected layer to match your number of classes
in_feat = model.fc.in_features
model.fc = nn.Linear(in_feat, len(class_names))
# Load checkpoint
model.load_state_dict(ckpt["model"])
model.to(device).eval()

tform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

# --- read image safely ---
img = cv2.imread(IMG)[:, :, ::-1]
x = tform(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(x); probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
pred = int(np.argmax(probs))
print("Prediction:", class_names[pred], " | Confidence:", float(probs[pred]))

# Grad-CAM
cam = gradcam_on_image(model, x, target_class=pred)
overlay = overlay_cam(img, cam)
cv2.imwrite("outputs/gradcam_overlay.jpg", overlay[:, :, ::-1])
print("Saved Grad-CAM to outputs/gradcam_overlay.jpg")
