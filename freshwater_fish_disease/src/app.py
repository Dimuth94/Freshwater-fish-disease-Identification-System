import io, cv2, torch
import numpy as np
import gradio as gr
from fastapi import FastAPI
from torchvision import models, transforms
import torch.nn as nn
from gradcam import gradcam_on_image, overlay_cam
from kb import TreatmentKB

# ---- load classifier ----
CKPT = "outputs/best.pt"
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT, map_location="cpu")
class_names = ckpt["classes"]

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_feat = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_feat, len(class_names))
model.load_state_dict(ckpt["model"])
model.to(device).eval()

tform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

kb = TreatmentKB("data/kb/treatments.csv", "data/kb/references.csv")

def diagnose(image, symptom_text):
    # image is PIL or numpy
    if image is None: return None, "No image", []
    if isinstance(image, np.ndarray):
        rgb = image
    else:
        rgb = np.array(image)

    x = tform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x); probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    pred = class_names[idx]; conf = float(probs[idx])

    # Grad-CAM overlay
    cam = gradcam_on_image(model, x, target_class=idx)
    overlay = overlay_cam(rgb, cam)

    # Retrieve treatments (match folder/class name to disease_name in KB)
    tr = kb.treatments_for(pred)
    treatments = tr.to_dict("records")

    caption = f"Prediction: {pred} (confidence {conf:.2f})\nSymptom text: {symptom_text or '-'}"
    return overlay, caption, treatments

with gr.Blocks() as demo:
    gr.Markdown("# AquaAid – Fish Disease Diagnosis (Demo)")
    with gr.Row():
        img = gr.Image(type="numpy", label="Upload fish image")
        txt = gr.Textbox(label="Symptom text (optional)", placeholder="e.g., white spots on tail, rapid breathing")
    btn = gr.Button("Diagnose")
    out_img = gr.Image(label="Grad-CAM focus")
    out_txt = gr.Textbox(label="Prediction")
    out_tbl = gr.Dataframe(headers=["disease_id","disease_name","synonyms","treatment_name","active_substance","dosage_guidance","cautions","source_link","citation"], wrap=True)
    btn.click(diagnose, inputs=[img, txt], outputs=[out_img, out_txt, out_tbl])

app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

# Run with:
# uvicorn src.app:app --reload --port 7860
