import io
import os
import uuid
import base64
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import torch
import torchvision.transforms as transforms
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

try:
    import torchxrayvision as xrv
except Exception as e:
    xrv = None
    print(f"[startup] torchxrayvision import skipped: {e}")

app = FastAPI(title="InnoTech AI Backend", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 1. Env-var-driven paths (works in Colab AND on Railway)
# ---------------------------------------------------------------------------
IN_COLAB = os.path.exists("/content/drive")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/content/drive/MyDrive/InnoTech_Dataset/ResNet50_merged_fixed.keras" if IN_COLAB
    else "./ResNet50_merged_fixed.keras",
)
if IN_COLAB and not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/content/drive/MyDrive/InnoTech_Dataset/Models_CNN/ResNet50_merged_fixed.keras"

RAG_CSV_PATH = os.getenv(
    "RAG_CSV_PATH",
    "/content/drive/MyDrive/InnoTech_Dataset/RAG/rag_data.csv" if IN_COLAB
    else "./rag_data.csv",
)
if IN_COLAB and not os.path.exists(RAG_CSV_PATH):
    RAG_CSV_PATH = "/content/drive/MyDrive/InnoTech_Dataset/rag_data.csv"

FAISS_INDEX_DIR = os.getenv(
    "FAISS_INDEX_DIR",
    "/content/drive/MyDrive/InnoTech_Dataset/RAG/faiss_index" if IN_COLAB
    else "./faiss_index",
)

# ---------------------------------------------------------------------------
# 2. Startup crash-proofing: X-ray models load defensively
# ---------------------------------------------------------------------------
model = None
base_grad_model = None
head_layers = []

try:
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        resnet_backbone = model.layers[0] if isinstance(model.layers[0], keras.Model) else model
        head_layers = model.layers[1:] if resnet_backbone != model else []
        target_conv_layer = resnet_backbone.get_layer("conv4_block6_out")
        base_grad_model = keras.Model(
            inputs=resnet_backbone.inputs,
            outputs=[target_conv_layer.output, resnet_backbone.output]
        )
    else:
        print(f"[startup] MODEL_PATH not found at {MODEL_PATH}; /predict will return 503.")
except Exception as e:
    print(f"[startup] X-ray classifier not loaded: {e}")

seg_device = "cuda" if torch.cuda.is_available() else "cpu"
seg_model = None

try:
    if xrv is not None:
        seg_model = xrv.baseline_models.chestx_det.PSPNet().to(seg_device).eval()
except Exception as e:
    print(f"[startup] Lung segmentation model not loaded: {e}")


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


# ---------------------------------------------------------------------------
# 3. RAG: embeddings + FAISS (crash-proofed: retriever=None if unavailable)
# ---------------------------------------------------------------------------
embedding_model = None
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
    )
except Exception as e:
    print(f"[startup] Embedding model not loaded: {e}")

vector_store = None
try:
    if embedding_model is not None:
        if os.path.exists(FAISS_INDEX_DIR):
            vector_store = FAISS.load_local(
                FAISS_INDEX_DIR,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        elif os.path.exists(RAG_CSV_PATH):
            df = pd.read_csv(RAG_CSV_PATH)
            df = df.dropna(subset=["text"]).copy()
            df["text"] = df["text"].astype(str).str.strip()
            df = df[df["text"] != ""].copy()
            rag_df = df.drop_duplicates(subset="text").copy()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = []
            for _, row in rag_df.iterrows():
                report_text = str(row["text"]).strip()
                uid_val = str(row["uid"]) if "uid" in rag_df.columns else "N/A"
                if report_text:
                    report_chunks = text_splitter.create_documents(
                        [report_text],
                        metadatas=[{"uid": uid_val}]
                    )
                    chunks.extend(report_chunks)
            if chunks:
                vector_store = FAISS.from_documents(chunks, embedding_model)
                try:
                    vector_store.save_local(FAISS_INDEX_DIR)
                except Exception:
                    pass
        else:
            print(
                f"[startup] No FAISS index at {FAISS_INDEX_DIR} and no CSV at {RAG_CSV_PATH}; "
                "/chat will answer without retrieved report context."
            )
except Exception as e:
    print(f"[startup] Vector store not loaded: {e}")

retriever = (
    vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    if vector_store is not None else None
)

# ---------------------------------------------------------------------------
# 4. Hosted LLM (Groq, OpenAI-compatible)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
llm_client = None

if GROQ_API_KEY:
    llm_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI clinical assistant for chest radiology and lung screening (such as LDCT eligibility and general pulmonary health).
Rules:
- If the user asks about specific radiology findings, rely strictly on the provided report context below.
- If the user asks general educational questions about lung screening (e.g., LDCT criteria, symptoms, follow-up guidelines), provide accurate, concise clinical guidance without prescribing medications or dosages.
- Never issue a definitive medical diagnosis; always remind that findings require physician review.
- Respond in the same language as the user's question (English or Arabic).
Retrieved Radiology Context:
{context}"""),
    ("human", "{question}")
])


def generate_rag_answer(question: str):
    if llm_client is None:
        raise RuntimeError(
            "LLM is not configured: set the GROQ_API_KEY environment variable "
            "(Railway -> your service -> Variables) and redeploy."
        )
    retrieved_docs = retriever.invoke(question) if retriever is not None else []
    context = (
        "\n\n".join(doc.page_content for doc in retrieved_docs)
        if retrieved_docs else "No specific matching report found."
    )
    messages = rag_prompt.format_messages(context=context, question=question)
    completion = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": messages[0].content},
            {"role": "user", "content": messages[1].content},
        ],
        max_tokens=250,
        temperature=0,
    )
    answer = completion.choices[0].message.content.strip()
    sources = [
        {"uid": doc.metadata.get("uid", "N/A"), "snippet": doc.page_content}
        for doc in retrieved_docs
    ]
    return answer, sources


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
    if model is None or base_grad_model is None or seg_model is None:
        raise HTTPException(status_code=503, detail="X-ray models are not loaded on this server.")
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
    visual_finding = (
        "Abnormal Opacity / Attention Detected"
        if score >= 0.5 else "Unremarkable / Clear Lung Fields"
    )

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
    try:
        answer, sources = generate_rag_answer(user_query)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[/chat] generate_rag_answer failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail=f"Chat backend error: {type(e).__name__}: {e}")

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
