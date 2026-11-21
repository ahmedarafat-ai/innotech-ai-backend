import io
import os
import uuid
import base64
import cv2
import gdown
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import torchxrayvision as xrv
import torchvision.transforms as transforms
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from groq import Groq

app = FastAPI(title="InnoTech AI Backend", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "ResNet50_merged_fixed.keras"
RAG_CSV_PATH = "rag_data.csv"

if not os.path.exists(MODEL_PATH):
    gdown.download(id="1dShO6i6sJ3rA-f8UcqB13zyI09Fp9bmp", output=MODEL_PATH, quiet=False)

if not os.path.exists(RAG_CSV_PATH):
    gdown.download(id="1s-EgmCDZoVYD3r3_DGGhCpXnHYntmbcq", output=RAG_CSV_PATH, quiet=False)

model = keras.models.load_model(MODEL_PATH)
resnet_backbone = model.layers[0] if isinstance(model.layers[0], keras.Model) else model
head_layers = model.layers[1:] if resnet_backbone != model else []
target_conv_layer = resnet_backbone.get_layer("conv4_block6_out")

base_grad_model = keras.Model(
    inputs=resnet_backbone.inputs,
    outputs=[target_conv_layer.output, resnet_backbone.output]
)

def compute_gradcam(img_tensor):
    img_tf = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tf)
        conv_outputs, backbone_output = base_grad_model(img_tf)
        tape.watch(conv_outputs)
        x = backbone_output
        for layer in head_layers:
            x = layer(x)
        score = x[0][0]

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        return np.zeros((7, 7), dtype=np.float32), float(score.numpy())

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap /= max_val
    return heatmap.numpy(), float(score.numpy())

seg_device = "cpu"
seg_model = xrv.baseline_models.chestx_det.PSPNet().to(seg_device).eval()

def compute_lung_segmentation(pil_img_rgb):
    gray_img = np.array(pil_img_rgb.convert("L"), dtype=np.float32)
    img_norm = xrv.datasets.normalize(gray_img, 255)[None, ...]
    transform = transforms.Compose([xrv.datasets.XRayResizer(512)])
    img_resized = transform(img_norm)
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).to(seg_device)

    with torch.no_grad():
        pred = seg_model(img_tensor)

    left_lung_idx = seg_model.targets.index("Left Lung")
    right_lung_idx = seg_model.targets.index("Right Lung")
    lung_prob = torch.sigmoid(pred[0, left_lung_idx]) + torch.sigmoid(pred[0, right_lung_idx])
    mask_512 = (lung_prob.cpu().numpy() > 0.5).astype(np.uint8) * 255

    mask_224 = cv2.resize(mask_512, (224, 224), interpolation=cv2.INTER_NEAREST)
    orig_224 = np.array(pil_img_rgb.resize((224, 224)), dtype=np.uint8)
    contours, _ = cv2.findContours(mask_224, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_overlay = orig_224.copy()
    cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 2)

    buf_mask = io.BytesIO()
    Image.fromarray(mask_224).save(buf_mask, format="PNG")
    mask_b64 = base64.b64encode(buf_mask.getvalue()).decode("utf-8")

    buf_overlay = io.BytesIO()
    Image.fromarray(contour_overlay).save(buf_overlay, format="PNG")
    overlay_b64 = base64.b64encode(buf_overlay.getvalue()).decode("utf-8")

    return mask_b64, overlay_b64

df = pd.read_csv(RAG_CSV_PATH).dropna(subset=["text"]).copy()
df["text"] = df["text"].astype(str).str.strip()
rag_df = df[df["text"] != ""].drop_duplicates(subset="text").reset_index(drop=True)

def retrieve_reports(query: str, k: int = 3):
    q_words = set(query.lower().split())
    if not q_words:
        return []
    scores = []
    for idx, row in rag_df.iterrows():
        text_lower = row["text"].lower()
        overlap = sum(1 for w in q_words if len(w) > 2 and w in text_lower)
        if overlap > 0:
            scores.append((overlap, idx))
    scores.sort(reverse=True)
    results = []
    for _, idx in scores[:k]:
        row = rag_df.iloc[idx]
        results.append({
            "uid": str(row["uid"]) if "uid" in rag_df.columns else "N/A",
            "snippet": row["text"][:500]
        })
    return results

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

def generate_rag_answer(question: str):
    sources = retrieve_reports(question, k=3)
    context = "\n\n".join(s["snippet"] for s in sources) if sources else "No specific matching report found."

    system_prompt = f"""You are an AI clinical assistant for chest radiology and lung screening (such as LDCT eligibility and general pulmonary health).
Rules:
- If the user asks about specific radiology findings, rely strictly on the provided report context below.
- If the user asks general educational questions about lung screening (e.g., LDCT criteria, symptoms, follow-up guidelines), provide accurate, concise clinical guidance without prescribing medications or dosages.
- Never issue a definitive medical diagnosis; always remind that findings require physician review.
- Respond in the same language as the user's question (English or Arabic).

Retrieved Radiology Context:
{context}"""

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return completion.choices[0].message.content, sources

chat_history_store = []

@app.get("/")
def root():
    return {"status": "online", "service": "InnoTech AI Backend", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

class LoginData(BaseModel):
    email: str
    password: str

def handle_login(data: LoginData):
    if data.email == "doctor@hospital.com" and data.password == "123456":
        token = f"innotech-jwt-{uuid.uuid4().hex}"
        return {
            "status": "success",
            "access_token": token,
            "token_type": "bearer",
            "user": {"email": data.email, "role": "physician"}
        }
    raise HTTPException(status_code=401, detail="Invalid email or password")

@app.post("/login")
def login(data: LoginData):
    return handle_login(data)

@app.post("/api/v1/auth/login")
def api_login(data: LoginData):
    return handle_login(data)

@app.post("/predict")
def predict_xray(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (PNG/JPG).")

    try:
        image_bytes = file.file.read()
        raw_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or unreadable image file.")

    pil_img = raw_pil.resize((224, 224))
    raw_img_array = np.array(pil_img, dtype=np.float32)

    img_for_model = np.expand_dims(raw_img_array.copy(), axis=0)
    img_preprocessed = preprocess_input(img_for_model)

    heatmap, score = compute_gradcam(img_preprocessed)
    visual_finding = "Abnormal Opacity / Attention Detected" if score >= 0.5 else "Unremarkable / Clear Lung Fields"

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.uint8(raw_img_array), 0.65, heatmap_colored, 0.35, 0)

    overlay_pil = Image.fromarray(overlay)
    buffered = io.BytesIO()
    overlay_pil.save(buffered, format="PNG")
    gradcam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    seg_mask_b64, seg_overlay_b64 = compute_lung_segmentation(raw_pil)

    return {
        "status": "success",
        "visual_finding": visual_finding,
        "gradcam_base64": gradcam_base64,
        "segmentation_mask_base64": seg_mask_b64,
        "segmentation_overlay_base64": seg_overlay_b64,
        "disclaimer": "AI visual decision-support output only. Not a definitive clinical diagnosis."
    }

class ChatRequest(BaseModel):
    question: Optional[str] = None
    message: Optional[str] = None

def process_chat(data: ChatRequest):
    user_query = (data.message or data.question or "").strip()
    if not user_query:
        raise HTTPException(status_code=422, detail="Either 'message' or 'question' is required.")

    answer, sources = generate_rag_answer(user_query)
    chat_history_store.append({"sender": "doctor", "message": user_query})
    chat_history_store.append({"sender": "ai", "message": answer})
    return {
        "status": "success",
        "question": user_query,
        "message": user_query,
        "answer": answer,
        "response": answer,
        "sources": sources
    }

@app.post("/chat")
def medical_chat(data: ChatRequest):
    return process_chat(data)

@app.post("/api/v1/ai/chat/messages")
def mobile_medical_chat(data: ChatRequest):
    return process_chat(data)

@app.get("/chat/history")
def get_chat_history():
    return {
        "status": "success",
        "history": chat_history_store
    }
