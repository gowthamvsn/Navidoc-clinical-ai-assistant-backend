# # FastAPI backend that exposes your CXR model as /api/cxr/predict
# # Run with: uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
# import os
# import shutil
# import warnings
# from typing import Optional
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, List, Tuple
# from io import BytesIO
# import torch
# import torch.nn.functional as F
# import torchxrayvision as xrv
# from PIL import Image, ImageEnhance
# import numpy as np

# warnings.filterwarnings("ignore", category=FutureWarning)
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# # -------------------------
# # CORS
# # -------------------------
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # tighten in prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------
# # Device / performance
# # -------------------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# AMP_ENABLED = torch.cuda.is_available()

# # -------------------------
# # Load model (domain-matched weights with fallback)
# # -------------------------
# def load_model():
#     try:
#         m = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")  # domain-matched
#     except Exception:
#         m = xrv.models.DenseNet(weights="densenet121-res224-chex")      # fallback
#     m.eval().to(DEVICE)
#     return m

# model = load_model()
# labels = model.pathologies

# # -------------------------
# # Temperature scaling (stub)
# # -------------------------
# SAVED_LOGT = 0.0  # replace with a value you learn offline, e.g., 0.15

# def calibrate_probs(p_vec, logT=SAVED_LOGT):
#     p = np.clip(np.asarray(p_vec), 1e-6, 1 - 1e-6)
#     logits = np.log(p / (1 - p))
#     logits = logits / np.exp(logT)
#     return 1 / (1 + np.exp(-logits))

# # -------------------------
# # Lay-language mapping
# # -------------------------
# LAY_TERMS = {
#     "Fracture": "a possible broken rib or collarbone",
#     "Lung Lesion": "a spot in the lung",
#     "Lung Opacity": "a cloudy area in the lung",
#     "Atelectasis": "part of the lung not fully open",
#     "Enlarged Cardiomediastinum": "the center of the chest looks wider than usual",
#     "Cardiomegaly": "the heart looks larger than usual",
#     "Effusion": "fluid around the lung",
#     "Pneumonia": "signs of a lung infection",
#     "Nodule": "a small round spot",
#     "Mass": "a larger lump",
#     "Infiltration": "patchy changes in the lung",
#     "Edema": "extra fluid in the lungs",
#     "Pneumothorax": "air outside the lung causing it to partly collapse",
#     "Consolidation": "a solid-looking area in the lung",
#     "Pleural_Thickening": "thickening of the lining around the lung",
# }
# def to_lay(lbl: str) -> str:
#     return LAY_TERMS.get(lbl, lbl.replace("_", " ").lower())

# # -------------------------
# # Smart crop → preprocess
# # -------------------------
# def smart_lung_crop(img_gray: Image.Image) -> Image.Image:
#     arr = np.array(img_gray)
#     q_low, q_high = np.quantile(arr, [0.05, 0.95])
#     mask = (arr >= q_low) & (arr <= q_high)
#     ys, xs = np.where(mask)
#     if ys.size:
#         y0, y1 = ys.min(), ys.max()
#         x0, x1 = xs.min(), xs.max()
#         py, px = int(0.05 * (y1 - y0 + 1)), int(0.05 * (x1 - x0 + 1))
#         y0 = max(0, y0 - py); y1 = min(arr.shape[0], y1 + py)
#         x0 = max(0, x0 - px); x1 = min(arr.shape[1], x1 + px)
#         arr = arr[y0:y1, x0:x1]
#     else:
#         h, w = arr.shape
#         dh, dw = int(h*0.05), int(w*0.05)
#         arr = arr[dh:h-dh, dw:w-dw]
#     return Image.fromarray(arr)

# def preprocess(pil_img: Image.Image) -> torch.Tensor:
#     g = pil_img.convert("L")
#     g = smart_lung_crop(g)
#     arr = np.array(g)
#     arr = xrv.datasets.normalize(arr, 255)
#     arr = np.expand_dims(arr, axis=(0, 1))
#     return torch.from_numpy(arr).float().to(DEVICE)

# # -------------------------
# # TTA predictions
# # -------------------------
# def tta_predictions(pil_img: Image.Image) -> np.ndarray:
#     aug_fns = [
#         lambda im: im,
#         lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
#         lambda im: ImageEnhance.Brightness(im).enhance(1.08),
#         lambda im: ImageEnhance.Contrast(im).enhance(1.08),
#     ]
#     preds = []
#     for fn in aug_fns:
#         t = preprocess(fn(pil_img))
#         with torch.no_grad():
#             with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
#                 out = model(t)
#             out_np = out[0].detach().cpu().numpy()
#             if (out_np.max() > 1.0) or (out_np.min() < 0.0):
#                 out_np = 1 / (1 + np.exp(-out_np))
#             preds.append(out_np)
#     p = np.mean(preds, axis=0)
#     return calibrate_probs(p, SAVED_LOGT)

# # -------------------------
# # Confidence summary
# # -------------------------
# def interpret_confidence(results: Dict[str, float], k: int = 4) -> str:
#     if not results: return "The model isn’t sure about this image."
#     items = sorted(results.items(), key=lambda x: x[1], reverse=True)
#     (top_label, top_p), second_p = items[0], (items[1][1] if len(items) > 1 else 0.0)
#     margin = top_p - second_p

#     def list_lay(itms, start=0, end=3):
#         terms = [to_lay(lbl) for lbl, _ in itms[start:end]]
#         if len(terms) >= 2: return ", ".join(terms[:-1]) + " or " + terms[-1]
#         return terms[0] if terms else ""

#     if top_p >= 0.85 and margin >= 0.20:
#         return f"The model is very confident the image may show {to_lay(top_label)}."
#     elif top_p >= 0.70 and margin >= 0.10:
#         return f"The model leans toward {to_lay(top_label)}, with other options less likely."
#     elif top_p >= 0.50 and margin < 0.10:
#         return f"The model cannot choose one clearly. {to_lay(top_label)} looks about as likely as {list_lay(items,1,3)}."
#     else:
#         return f"The model’s confidence is low and spread across a few possibilities: {list_lay(items,0,3)}."

# DISCLAIMER = "This AI output is not a medical diagnosis. Please consult a clinician for decisions."

# class CXRResponse(BaseModel):
#     top5: List[Tuple[str, float]]
#     summary: str

# @app.get("/healthz")
# def healthz():
#     return {"ok": True}

# @app.post("/api/cxr/predict", response_model=CXRResponse)
# async def predict(image: UploadFile = File(...)):
#     raw = await image.read()
#     pil = Image.open(BytesIO(raw)).convert("RGB")

#     probs = tta_predictions(pil)  # [C]
#     results = {lbl: float(probs[i]) for i, lbl in enumerate(labels)}
#     top5 = sorted(results.items(), key=lambda kv: kv[1], reverse=True)[:5]

#     conf_text = interpret_confidence(results)
#     top_sentence = ", ".join([f"{to_lay(lbl)} ({prob*100:.0f}%)" for lbl, prob in top5])
#     summary = f"Summary: {conf_text} Top possibilities based on the image: {top_sentence}.  \n\n_{DISCLAIMER}_"

#     return {"top5": top5, "summary": summary}





# main.py
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
import time
from io import BytesIO
import torch
import torchxrayvision as xrv
from PIL import Image, ImageEnhance

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import re
import sys
import time
import json
import hashlib
import warnings
from typing import Optional, Tuple, Dict, Any, List, Union
import pandas as pd
import numpy as np
import certifi

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------
# Optional deps with helpful errors
# ----------------------
try:
    from pymongo import MongoClient
    from pymongo.operations import SearchIndexModel
except Exception:
    print("ERROR: pymongo not installed. pip install pymongo", file=sys.stderr)
    raise

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    print("ERROR: sentence-transformers not installed. pip install sentence-transformers", file=sys.stderr)
    raise

try:
    from sqlalchemy import create_engine, text as sa_text
except Exception:
    print("ERROR: SQLAlchemy not installed. pip install SQLAlchemy psycopg2-binary", file=sys.stderr)
    raise

# LangChain is optional; degrade gracefully if missing
try:
    from langchain_community.utilities import SQLDatabase
    from langchain.chains import create_sql_query_chain
    from langchain_experimental.sql import SQLDatabaseChain
    LANGCHAIN_OK = True
except Exception:
    LANGCHAIN_OK = False

try:
    from langchain_openai import AzureChatOpenAI
    LC_OPENAI_OK = True
except Exception:
    LC_OPENAI_OK = False

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
except Exception:
    pass

# ----------------------
# Env & config helpers
# ----------------------
def require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        sys.exit(f"Missing env var: {name}. Set it in .env and reload.")
    return v

# ----------------------
# Intent (Gemini) setup
# ----------------------
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except Exception:
    print("ERROR: google-generativeai not installed. pip install google-generativeai", file=sys.stderr)
    raise

GOOGLE_API_KEY = require("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
_GEMINI = None
for m in ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro']:
    try:
        _GEMINI = genai.GenerativeModel(m)
        break
    except google_exceptions.InvalidArgument:
        continue
if _GEMINI is None:
    sys.exit("No compatible Gemini model available.")

AZURE_OPENAI_API_KEY    = require("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT   = require("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = require("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

DB_HOST = require("DB_HOST")
DB_USER = require("DB_USER")
DB_PASSWORD = require("DB_PASSWORD")
DB_NAME = require("DB_NAME")

SEARCH_PATH = os.getenv("SEARCH_PATH", "mimiciv_hosp,mimiciv_icu,public")

# MongoDB Atlas configuration (SQL templates)
MONGODB_URI = require("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "mimic_sql_assistant")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "sql_queries")
MONGODB_INDEX_NAME = os.getenv("MONGODB_INDEX_NAME", "vector_index")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

SQL_LARGE_TABLES  = set(os.getenv("SQL_LARGE_TABLES", "chartevents,labevents").split(","))
PRINT_SQL = os.getenv("PRINT_SQL", "true").lower() == "true"
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.70"))

# Postgres engine
uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"
engine = create_engine(
    uri,
    pool_pre_ping=True,
    connect_args={"options": f"-csearch_path={SEARCH_PATH}"},
)

# LangChain (optional)
if LANGCHAIN_OK and LC_OPENAI_OK:
    db = SQLDatabase.from_uri(
        uri,
        engine_args={"connect_args": {"options": f"-csearch_path={SEARCH_PATH}"}},
        include_tables={
            "patients", "admissions", "transfers",
            "diagnoses_icd", "d_icd_diagnoses",
            "procedures_icd", "d_icd_procedures",
            "labevents", "d_labitems",
            "prescriptions",
            "microbiologyevents",
            "icustays", "chartevents", "d_items",
        },
        sample_rows_in_table_info=2
    )
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VER,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        temperature=0.0,
        max_tokens=300,
    )
    try:
        sql_gen_chain = create_sql_query_chain(llm, db)
    except Exception:
        sql_gen_chain = None
    try:
        sql_db_chain  = SQLDatabaseChain.from_llm(
            llm=llm, db=db, verbose=False, top_k=10, return_intermediate_steps=True
        )
    except Exception:
        sql_db_chain = None
else:
    db = None
    llm = None
    sql_gen_chain = None
    sql_db_chain = None

# ---- Helper to make stable Mongo clients (Certifi CA) ----
def make_mongo_client(uri: str) -> MongoClient:
    return MongoClient(
        uri,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=30000,
        socketTimeoutMS=30000,
        connectTimeoutMS=30000,
        retryWrites=True,
    )

# MongoDB for SQL templates
mongo_client = make_mongo_client(MONGODB_URI)
mongo_db = mongo_client[MONGODB_DB_NAME]
collection = mongo_db[MONGODB_COLLECTION_NAME]

embedder = SentenceTransformer(EMBEDDING_MODEL)
EMBEDDING_DIM = embedder.get_sentence_embedding_dimension()

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# Initialize SQL template store
try:
    mongo_client.admin.command("ping")
except Exception:
    pass


# ============================================================
# ======================= FastAPI Setup ======================
# ============================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ===================== CXR MODEL PIPELINE ===================
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = torch.cuda.is_available()

def load_model():
    try:
        m = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    except Exception:
        m = xrv.models.DenseNet(weights="densenet121-res224-chex")
    m.eval().to(DEVICE)
    return m

model = load_model()
labels = model.pathologies
SAVED_LOGT = 0.0

def calibrate_probs(p_vec, logT=SAVED_LOGT):
    p = np.clip(np.asarray(p_vec), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p))
    logits = logits / np.exp(logT)
    return 1 / (1 + np.exp(-logits))

LAY_TERMS = {
    "Fracture": "a possible broken rib or collarbone",
    "Lung Lesion": "a spot in the lung",
    "Lung Opacity": "a cloudy area in the lung",
    "Atelectasis": "part of the lung not fully open",
    "Enlarged Cardiomediastinum": "the center of the chest looks wider than usual",
    "Cardiomegaly": "the heart looks larger than usual",
    "Effusion": "fluid around the lung",
    "Pneumonia": "signs of a lung infection",
    "Nodule": "a small round spot",
    "Mass": "a larger lump",
    "Infiltration": "patchy changes in the lung",
    "Edema": "extra fluid in the lungs",
    "Pneumothorax": "air outside the lung causing it to partly collapse",
    "Consolidation": "a solid-looking area in the lung",
    "Pleural_Thickening": "thickening of the lining around the lung",
}
def to_lay(lbl: str) -> str:
    return LAY_TERMS.get(lbl, lbl.replace("_", " ").lower())

def smart_lung_crop(img_gray: Image.Image) -> Image.Image:
    arr = np.array(img_gray)
    q_low, q_high = np.quantile(arr, [0.05, 0.95])
    mask = (arr >= q_low) & (arr <= q_high)
    ys, xs = np.where(mask)
    if ys.size:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        py, px = int(0.05 * (y1 - y0 + 1)), int(0.05 * (x1 - x0 + 1))
        y0 = max(0, y0 - py); y1 = min(arr.shape[0], y1 + py)
        x0 = max(0, x0 - px); x1 = min(arr.shape[1], x1 + px)
        arr = arr[y0:y1, x0:x1]
    else:
        h, w = arr.shape
        dh, dw = int(h*0.05), int(w*0.05)
        arr = arr[dh:h-dh, dw:w-dw]
    return Image.fromarray(arr)

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    g = pil_img.convert("L")
    g = smart_lung_crop(g)
    arr = np.array(g)
    arr = xrv.datasets.normalize(arr, 255)
    arr = np.expand_dims(arr, axis=(0, 1))
    return torch.from_numpy(arr).float().to(DEVICE)

def tta_predictions(pil_img: Image.Image) -> np.ndarray:
    aug_fns = [
        lambda im: im,
        lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        lambda im: ImageEnhance.Brightness(im).enhance(1.08),
        lambda im: ImageEnhance.Contrast(im).enhance(1.08),
    ]
    preds = []
    for fn in aug_fns:
        t = preprocess(fn(pil_img))
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                out = model(t)
            out_np = out[0].detach().cpu().numpy()
            if (out_np.max() > 1.0) or (out_np.min() < 0.0):
                out_np = 1 / (1 + np.exp(-out_np))
            preds.append(out_np)
    return calibrate_probs(np.mean(preds, axis=0), SAVED_LOGT)

def interpret_confidence(results: Dict[str, float]) -> str:
    if not results: return "The model isn’t sure about this image."
    items = sorted(results.items(), key=lambda x: x[1], reverse=True)
    (top_label, top_p), second_p = items[0], (items[1][1] if len(items) > 1 else 0.0)
    margin = top_p - second_p

    def list_lay(itms, start=0, end=3):
        terms = [to_lay(lbl) for lbl, _ in itms[start:end]]
        if len(terms) >= 2: return ", ".join(terms[:-1]) + " or " + terms[-1]
        return terms[0] if terms else ""

    if top_p >= 0.85 and margin >= 0.20:
        return f"The model is very confident the image may show {to_lay(top_label)}."
    elif top_p >= 0.70 and margin >= 0.10:
        return f"The model leans toward {to_lay(top_label)}, with other options less likely."
    elif top_p >= 0.50 and margin < 0.10:
        return f"The model cannot choose one clearly. {to_lay(top_label)} looks about as likely as {list_lay(items,1,3)}."
    else:
        return f"The model’s confidence is low and spread across a few possibilities: {list_lay(items,0,3)}."

DISCLAIMER = "This AI output is not a medical diagnosis. Please consult a clinician for decisions."

def analyze_cxr_pil(pil: Image.Image) -> Dict[str, Any]:
    probs = tta_predictions(pil)
    results = {lbl: float(probs[i]) for i, lbl in enumerate(labels)}
    top5 = sorted(results.items(), key=lambda kv: kv[1], reverse=True)[:5]
    conf_text = interpret_confidence(results)
    top_sentence = ", ".join([f"{to_lay(lbl)} ({prob*100:.0f}%)" for lbl, prob in top5])
    summary = f"Summary: {conf_text} Top possibilities based on the image: {top_sentence}.  \n\n_{DISCLAIMER}_"
    return {"top5": top5, "summary": summary}




# Diagnose 
LOGIC_URI  = require("logic_MONGODB_URI")
LOGIC_DB   = require("logic_DB_NAME")
LOGIC_COLL = require("logic_COLLECTION_NAME")
LOGIC_INDEX_NAME = os.getenv("LOGIC_VECTOR_INDEX_NAME")  # optional
LOGIC_EMBEDDING_MODEL = os.getenv("LOGIC_EMBEDDING_MODEL", EMBEDDING_MODEL)



# DB_HOST = require("DB_HOST"); DB_USER=require("DB_USER"); DB_PASSWORD=require("DB_PASSWORD"); DB_NAME=require("DB_NAME")


logic_client = make_mongo_client(LOGIC_URI)
logic_db = logic_client[LOGIC_DB]
logic_collection = logic_db[LOGIC_COLL]
try:
    logic_client.admin.command("ping")
except Exception:
    pass

logic_encoder = SentenceTransformer(LOGIC_EMBEDDING_MODEL)





def execute_sql(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(sa_text(sql), conn)
    
# Guards / utilities
def _strip_sql_comments(sql: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", sql, flags=re.S)
    s = re.sub(r"--[^\n]*", "", s)
    return s

def is_readonly_sql(sql: str) -> bool:
    s = _strip_sql_comments(sql).strip().lower()
    forbidden = r"\b(insert|update|delete|drop|alter|create|grant|revoke|truncate|comment|vacuum|copy|call|do|refresh|analyze)\b"
    if re.search(forbidden, s):
        return False
    return bool(re.match(r"^\s*(with|select|explain|show)\b", s))

def touches_large_tables(sql: str) -> bool:
    s = sql.lower()
    return any(re.search(rf"\b{re.escape(tbl)}\b", s) for tbl in SQL_LARGE_TABLES)

def has_aggregation(sql: str) -> bool:
    s = sql.lower()
    return bool(re.search(r"\b(count|sum|avg|min|max|median|percentile|stddev|variance)\s*\(", s))

def auto_limit(sql: str, limit: int = 200) -> str:
    s = sql.strip().rstrip(";")
    if re.search(r"\blimit\b\s+\d+", s, re.IGNORECASE):
        return s
    if has_aggregation(s):
        return s
    if touches_large_tables(s):
        return f"{s} LIMIT {limit}"
    return s


# ============================================================
# ===================== RAG + SQL PIPELINE ===================
# ============================================================

def classify_intent(user_question: str) -> str:
    """Return 'statistical' or 'reasoning' (default)."""
    prompt = f"""
        Classify this medical query as:
        - 'statistical': numbers, counts, averages, lowest/highest, totals, trends.
        - 'reasoning': patient, diagnosis, symptoms, causes, or advice without counting.

        Query: {user_question}
        Output ONLY 'statistical' or 'reasoning'.
    """
    try:
        resp = _GEMINI.generate_content(prompt)
        label = (resp.text or "").strip().lower()
        return label if label in ("statistical","reasoning") else "reasoning"
    except Exception:
        return "reasoning"

def retrieve_sql_template(question: str, k: int = 3) -> Tuple[Optional[str], float, List[Dict[str, Any]]]:
    print("Inside retrieve_sql_template question: ", question)
    try:
        query_embedding = embedder.encode(question).tolist()
        pipeline = [
            {"$vectorSearch": {
                "index": MONGODB_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": k * 10,
                "limit": k
            }},
            {"$match": {"_type": {"$ne": "metadata"}}},
            {"$project": {
                "question": 1, "sql_query": 1, "category": 1, "tables_used": 1,
                "score": {"$meta": "vectorSearchScore"}
            }}
        ]
        results = list(collection.aggregate(pipeline))
        if not results:
            return None, 0.0, []
        top = []
        for r in results:
            top.append({
                "id": str(r["_id"]),
                "sim": float(r.get("score", 0.0)),
                "question": r["question"],
                "sql_query": r["sql_query"],
                "category": r.get("category", ""),
                "tables_used": r.get("tables_used", "")
            })
        best = top[0]
        print("Inside retrieve_sql_template best SQL query: ", best["sql_query"])
        return best["sql_query"], best["sim"], top
    except Exception as e:
        print(f"[mongodb] Vector search failed: {e}")
        return None, 0.0, []
    

# LLM tuning / generation
SCHEMA_HINT = """
Use ONLY MIMIC-IV tables that are on the search_path:
- patients(subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod)
- admissions(subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, admission_location, discharge_location, insurance, language, marital_status, race, edregtime, edouttime, hospital_expire_flag)
- transfers(subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime)
- diagnoses_icd(subject_id, hadm_id, seq_num, icd_code, icd_version)
- d_icd_diagnoses(icd_code, icd_version, long_title)
- procedures_icd(subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version)
- d_icd_procedures(icd_code, icd_version, long_title)
- labevents(labevent_id, subject_id, hadm_id, specimen_id, itemid, charttime, storetime, value, valuenum, valueuom, ref_range_lower, ref_range_upper, flag, priority, comments)
- d_labitems(itemid, label, fluid, category)
- prescriptions(subject_id, hadm_id, pharmacy_id, poe_id, poe_seq, order_provider_id, starttime, stoptime, drug_type, drug, formulary_drug_cd, gsn, ndc, prod_strength, form_rx, dose_val_rx, dose_unit_rx, form_val_disp, form_unit_disp, doses_per_24_hrs, route)
- microbiologyevents(microevent_id, subject_id, hadm_id, micro_specimen_id, order_provider_id, chartdate, charttime, spec_itemid, spec_type_desc, test_seq, storedate, storetime, test_itemid, test_name, org_itemid, org_name, isolate_num, quantity, ab_itemid, ab_name, dilution_text, dilution_comparison, dilution_value, interpretation)
- icustays(subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los)
- chartevents(subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, valuenum, valueuom, warning)
- d_items(itemid, label, abbreviation, linksto, category, unitname, param_type, lownormalvalue, highnormalvalue)
"""

AGE_RULE = """Estimate age as: p.anchor_age + EXTRACT(YEAR FROM event_time)::int - p.anchor_year (do NOT reference date of birth)."""
UNIT_RULE = """If the user says "ICU" with no specific unit, use i.first_careunit ILIKE '%ICU%'. For specific units, use exact values such as 'Surgical Intensive Care Unit (SICU)', 'Medical Intensive Care Unit (MICU)', 'Coronary Care Unit (CCU)', 'Trauma SICU (TSICU)', 'Cardiac Vascular Intensive Care Unit (CVICU)'. """


def tune_sql(base_sql: str, user_q: str) -> str:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VER,
        proxies=None
    )
    prompt = f"""
        You are a PostgreSQL expert working with the MIMIC-IV schema on the current search_path.

        User question:
        {user_q}

        Base SQL (may be close but needs adaptation to exactly match the question):
        {base_sql}

        Requirements:
        {SCHEMA_HINT}
        {AGE_RULE}
        {UNIT_RULE}
        - Keep valid PostgreSQL syntax.
        - If the question asks for "top N", adjust the LIMIT or aggregation accordingly (e.g., top 7 diagnoses).
        - If the question changes disease/condition/unit/time-window, reflect that precisely in WHERE/JOIN conditions.
        - Prefer admissions.admittime or icustays.intime as event_time when you need a timestamp for age or time-based filters.
        - If you query chartevents or labevents without aggregation, include a LIMIT 200 to keep results small.
        - Return ONLY the final SQL. No explanations, no backticks.
    """
    r = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=700,
    )
    return (r.choices[0].message.content or "").strip()

def generate_sql_with_langchain(user_q: str) -> str:
    print("Inside generate_sql_with_langchain question: ", user_q)
    if 'sql_gen_chain' in globals() and sql_gen_chain is not None:
        try:
            candidate = sql_gen_chain.invoke({"question": user_q})
            if isinstance(candidate, str):
                return candidate
            if isinstance(candidate, dict):
                return candidate.get("query") or candidate.get("sql") or str(candidate)
            return str(candidate)
        except Exception:
            pass
    return "SELECT a.hadm_id, p.subject_id FROM admissions a JOIN patients p USING(subject_id) LIMIT 50"

def answer_statistical(question: str) -> Tuple[pd.DataFrame, str]:
    print("Inside statistical question: ", question)
    """Returns (dataframe, executed_sql, retrieved_similarity)."""
    retrieved_sql, sim, _top = retrieve_sql_template(question, k=3)

    if retrieved_sql and sim >= SIM_THRESHOLD:
        base_sql = retrieved_sql
    else:
        base_sql = generate_sql_with_langchain(question)

    tuned_sql = tune_sql(base_sql, question)
    print("answer_statistical tuned_sql: ", tuned_sql)
    tuned_sql = auto_limit(tuned_sql)

    df = None
    try:
        df = execute_sql(tuned_sql)
        print(" answer_statistical df: ", df)
    except Exception:
        pass

    if (df is None or df.empty) and 'sql_db_chain' in globals() and sql_db_chain is not None:
        try:
            res = sql_db_chain.invoke({"query": question})
            alt_sql = ""
            try:
                steps = res.get("intermediate_steps", []) if isinstance(res, dict) else []
            except Exception:
                steps = []
            for step in reversed(steps):
                if isinstance(step, dict):
                    alt_sql = step.get("sql_query") or step.get("sql") or alt_sql
                elif isinstance(step, str) and ("select " in step.lower() or " with " in step.lower()):
                    alt_sql = step
                if alt_sql:
                    break
            if alt_sql:
                alt_sql = auto_limit(alt_sql)
                try:
                    df2 = execute_sql(alt_sql)
                    if df2 is not None and not df2.empty:
                        tuned_sql = alt_sql
                        df = df2
                except Exception:
                    pass
        except Exception:
            pass

    # if df is not None and not df.empty:
    #     maybe_learn_template(question, tuned_sql, df, CSV_PATH)
    print("answer_statistical responsedf, tuend_sql: ", (df if df is not None else pd.DataFrame()), tuned_sql)
    return (df if df is not None else pd.DataFrame()), tuned_sql, (sim if sim else 0.0)

def logic_vector_search(query_text: str, n_results: int = 5) -> pd.DataFrame:
    """Vector-first with retry; degrades to sample/find if aggregation fails."""
    qvec = logic_encoder.encode(query_text).tolist()

    def _try_aggregate():
        idx_names = [LOGIC_INDEX_NAME] if LOGIC_INDEX_NAME else [
            "default","vector_index","vector_search_index","embedding_index"
        ]
        for idx in [n for n in idx_names if n]:
            try:
                pipeline = [
                    {"$vectorSearch": {
                        "queryVector": qvec,
                        "path": "embedding",
                        "numCandidates": 200,
                        "limit": n_results,
                        "index": idx
                    }},
                    {"$project": {
                        "hadm_id": 1, "narrative": 1, "subject_id": 1, "insurance": 1,
                        "age": 1, "diagnoses": 1, "abnormal_labs": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }}
                ]
                tmp = list(logic_collection.aggregate(pipeline))
                if tmp:
                    return tmp
            except Exception:
                continue
        return []

    results: List[Dict[str, Any]] = _try_aggregate()
    if not results:
        time.sleep(0.3)
        results = _try_aggregate()

    if not results:
        # fallback sampling
        try:
            results = list(logic_collection.aggregate([
                {"$sample": {"size": n_results}},
                {"$project": {
                    "hadm_id": 1,"narrative": 1,"subject_id": 1,"insurance": 1,
                    "age": 1,"diagnoses": 1,"abnormal_labs": 1
                }}
            ]))
        except Exception:
            try:
                results = list(logic_collection.find(
                    {}, {"hadm_id":1,"narrative":1,"subject_id":1,"insurance":1,"age":1,"diagnoses":1,"abnormal_labs":1}
                ).limit(n_results))
            except Exception:
                results = []

    rows = []
    for r in results:
        rows.append({
            "hadm_id": r.get("hadm_id",""),
            "narrative": r.get("narrative",""),
            "diagnoses": r.get("diagnoses","No diagnoses recorded."),
            "abnormal_labs": r.get("abnormal_labs","No labs found."),
            "subject_id": r.get("subject_id","unknown"),
            "age": r.get("age","unknown"),
            "insurance": r.get("insurance","unknown"),
        })
    return pd.DataFrame(rows)

# Reasoning (multi-agent)
def multi_agent_debate(user_question: str) -> str:
    df = logic_vector_search(user_question, n_results=5)
    if df is None or df.empty:
        return "No similar cases were retrieved from the database."

    parts = []
    for i, row in df.iterrows():
        parts.append(
            f"""
                PATIENT {i+1} (ID: {row['hadm_id']}):
                - Age: {row.get('age','?')}
                - Diagnoses: {row['diagnoses']}
                - Abnormal Labs: {row['abnormal_labs']}
                - Clinical Notes: {row['narrative'][:500]}...
            """.strip()
        )
    context = "\n".join(parts)

    debate_prompt = f"""
        MEDICAL CASE ANALYSIS

        Patient's Current Symptoms / Question: {user_question}

        Similar Cases from Medical Database:
        {context}

        You are three specialist doctors:

        🩺 Dr. Harrison (Diagnostician): pattern-match to likely conditions; cite Patient IDs; confidence.
        🏥 Dr. Chen (Differential): consider mimics and less obvious causes; cite IDs; confidence.
        📋 Dr. Rodriguez (Clinical): cross-check clusters, progression, responses; cite IDs; confidence.

        Then provide a CONSENSUS with top-2 diagnoses or key takeaways.

        Return a clear, well-structured analysis ONLY (no introductions, no extra prefaces).
    """
    try:
        r = _GEMINI.generate_content(debate_prompt)
        return (r.text or "").strip() or "Analysis unavailable."
    except Exception:
        return "Analysis unavailable."

def neutral_summary_from_debate(debate_text: str, user_question: str) -> str:
    """Concise, neutral synthesis; no advice, no 'you', no prefaces."""
    prompt = f"""
        Question: {user_question}

        Based on the clinician debate below, write a concise, neutral summary (facts only).
        Rules:
        - No second-person language (avoid 'you').
        - No recommendations, instructions, or clinical advice.
        - No prefaces like 'Here is', 'In summary', etc.
        - No disclaimers.
        - Prefer bullet points; <= 8 bullets or a short paragraph (<120 words).

        Debate:
        {debate_text}
    """
    try:
        t = _GEMINI.generate_content(prompt).text or ""
        summary = t.strip()
        summary = re.sub(r"^(here (is|are)\b|in summary\b|summary:\s*)", "", summary, flags=re.I).strip()
        return summary
    except Exception:
        return "Key points extracted from retrieved cases were unavailable."

def _parse_json_safely(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'(\{.*\}|\[.*\])', text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

def _format_summary(dxs: list) -> str:
    if not dxs:
        return "No high-confidence differentials identified from retrieved cases."
    lines = ["Possible diagnoses (from similar cases):"]
    for d in dxs:
        name = d.get("name","Condition")
        conf = d.get("confidence", None)
        conf_pct = f"{round(conf*100):d}%" if isinstance(conf,(int,float)) else "—"
        ids = d.get("patient_ids", []) or []
        ev = d.get("evidence","").strip()
        cite = f" [IDs: {', '.join(map(str, ids))}]" if ids else ""
        if ev and len(ev) > 240:
            ev = ev[:237] + "..."
        lines.append(f"• {name} — confidence {conf_pct}{cite}. {ev}")
    return "\n".join(lines)

def _build_case_context(df: pd.DataFrame) -> str:
    parts = []
    for i, row in df.iterrows():
        parts.append(
            f"""
            PATIENT {i+1} (ID: {row.get('hadm_id','')}):
            - Age: {row.get('age','?')}
            - Diagnoses: {row.get('diagnoses','')}
            - Abnormal Labs: {row.get('abnormal_labs','')}
            - Clinical Notes: {str(row.get('narrative',''))[:500]}...
            """.strip()
        )
    return "\n".join(parts)

# def diagnose(user_symptoms: str) -> dict:
#     r = _GEMINI.generate_content(f"Possible diagnoses for: {user_symptoms}")
#     return {"summary_text": (r.text or "").strip()}
# def diagnose(user_symptoms: Optional[str] = None, pil_img: Optional[Image.Image] = None, n_results: int = 5) -> Dict[str, Any]:
def diagnose(user_symptoms: Optional[str] = None, n_results: int = 5) -> Dict[str, Any]:
    # results: Dict[str, Any] = {}
    # # ---- TEXT BRANCH ----
    # if user_symptoms:
    #     r = _GEMINI.generate_content(f"Possible diagnoses for: {user_symptoms}")
    #     results["text"] = {"summary_text": (r.text or "").strip()}
    # # ---- IMAGE BRANCH ---- (reusing analyze_cxr_pil)
    # if pil_img:
    #     results["image"] = analyze_cxr_pil(pil_img)
    # return results
    """Diagnose pipeline for the 'Diagnose' button. No prints."""
    df = logic_vector_search(user_symptoms, n_results=n_results)
    cases_used = df.to_dict(orient="records") if (df is not None and not df.empty) else []
    context = _build_case_context(df) if cases_used else "No similar cases were retrieved from the database."

    diag_prompt = f"""
        A patient reports: "{user_symptoms}"

        Similar cases from the internal database:
        {context}

        TASK:
        - Identify the top 3 most likely diagnoses suggested by the retrieved cases.
        - For each, provide:
        * name: concise condition name
        * confidence: a number 0..1 estimating likelihood vs other listed items
        * patient_ids: the case IDs that support it
        * evidence: 1–2 sentence rationale citing concrete overlaps (symptoms, labs, narratives)

        STRICT OUTPUT (JSON only):
        {{
        "diagnoses": [
            {{"name": "...", "confidence": 0.0, "patient_ids": ["..."], "evidence": "..."}},
            ...
        ]
        }}
        No advice, no recommendations, no prefaces. JSON only.
    """
    try:
        r = _GEMINI.generate_content(diag_prompt)
        raw = (r.text or "").strip()
    except Exception:
        raw = ""

    parsed = _parse_json_safely(raw) or {}
    dxs = parsed.get("diagnoses", []) if isinstance(parsed, dict) else []

    norm = []
    for d in dxs:
        try:
            name = str(d.get("name","")).strip()[:120]
            conf = d.get("confidence", None)
            try:
                conf = float(conf)
                if not (0.0 <= conf <= 1.0):
                    conf = None
            except Exception:
                conf = None
            pids = d.get("patient_ids", [])
            if isinstance(pids, (str, int)):
                pids = [str(pids)]
            elif isinstance(pids, list):
                pids = [str(x) for x in pids][:6]
            else:
                pids = []
            ev = str(d.get("evidence","")).strip()
            norm.append({"name": name, "confidence": conf, "patient_ids": pids, "evidence": ev})
        except Exception:
            continue

    summary_text = _format_summary(norm)

    return {
        "intent": "diagnose",
        "cases_used": cases_used,
        "diagnoses": norm,
        "summary_text": summary_text,
        "raw_model_text": raw,
    }

def research(user_question: str):
    intent = classify_intent(user_question)
    print(f"INTENT: {'STATISTICAL' if intent=='statistical' else 'LOGICAL'}")
    result: Dict[str, Any] = {"intent": intent}

    if intent == "statistical":
        df, used_sql, sim = answer_statistical(user_question)
        print("\n=== SIMILARITY ===")
        print(f"{sim:.3f}")
        print("\n=== SQL USED ===")
        print(used_sql)
        print("\n=== RESULTS ===")
        if df is not None and not df.empty:
            try:
                pd.set_option("display.max_rows", 30)
                pd.set_option("display.max_columns", 20)
                print(df.to_string(index=False))
            except Exception:
                print(df.head(30))
        else:
            print("No rows returned.")
        result.update({
            "sql": used_sql,
            "results": df.to_dict(orient="records") if df is not None else []
        })
    else:
        debate = multi_agent_debate(user_question)
        neutral = neutral_summary_from_debate(debate, user_question)
        result.update({
            "summary": neutral
        })
    return result

# -------------------------
# Define request model for JSON
# -------------------------
class ResearchRequest(BaseModel):
    query: Optional[str] = None

# ============================================================
# ===================== API ROUTES ===========================
# ============================================================
class CXRResponse(BaseModel):
    top5: List[Tuple[str, float]]
    summary: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/api/cxr/predict", response_model=CXRResponse)
async def cxr_predict(image: UploadFile = File(...)):
    raw = await image.read()
    pil = Image.open(BytesIO(raw)).convert("RGB")
    return analyze_cxr_pil(pil)

@app.post("/api/research")
async def api_research(
    query: Optional[str] = Form(None),
    image: Union[UploadFile, None, str] = File(None),):
    # Normalize: Swagger sends "" when no file is uploaded
    if isinstance(image, str) or (image and getattr(image, "filename", "") == ""):
        image = None

    print(">>> [Research] Query:", query)
    print(">>> [Research] Image:", image.filename if image else None)

    if not query and not image:
        return {"error": "No input provided"}

    if query and not image:
        res = research(query)
        return {"intent": res["intent"], "result": res}

    if query and image:
        res = research(query)
        return {
            "intent": res["intent"],
            "result": res,
            "note": "Image uploaded — please use Diagnose button for image analysis."
        }

    if image and not query:
        return {"error": "For image input, please use the Diagnose button."}


@app.post("/api/diagnose")
async def api_diagnose(
    query: Optional[str] = Form(None),
    image: Union[UploadFile, None, str] = File(None),):
    # Normalize: Swagger sends "" when no file is uploaded
    if isinstance(image, str) or (image and getattr(image, "filename", "") == ""):
        image = None

    print(">>> [Diagnose] Query:", query)
    print(">>> [Diagnose] Image:", image.filename if image else None)
    """
    Diagnose endpoint:
    - If only text: run RAG Diagnose
    - If only image: run CXR model
    - If both: run both and return combined
    """
    if not query and not image:
        return {"error": "No input provided."}

    pil = None
    if image and not query:
        raw = await image.read()
        pil = Image.open(BytesIO(raw)).convert("RGB")
        return analyze_cxr_pil(pil)
    return diagnose(user_symptoms=query, pil_img=pil)
