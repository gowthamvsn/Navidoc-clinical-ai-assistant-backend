# Navidoc-clinical-ai-assistant-backend

A safety-first clinical AI backend that combines medical image analysis, database-backed clinical research, and retrieval-augmented reasoning.  
Designed strictly for **clinical decision support and research**, not diagnosis or treatment.

---

## ⚠️ Medical Disclaimer

This system **does not provide medical advice, diagnosis, or treatment**.  
All outputs are informational and must be interpreted by qualified healthcare professionals.

---

## 🎯 Motivation

Most healthcare AI systems suffer from one of two issues:
- Black-box predictions with no transparency
- Unsafe diagnostic claims without grounding in real clinical data

This project addresses both by:
- Using **domain-trained medical models**
- Querying **real hospital data (MIMIC-IV)**
- Enforcing **read-only SQL**
- Providing **evidence-based reasoning**
- Clearly separating *analysis* from *decision-making*

---

## 🧠 System Overview

### 1️⃣ Chest X-Ray Analysis (CXR)
- DenseNet-121 pretrained on MIMIC-CXR / CheXpert
- Smart lung cropping
- Test-time augmentation (TTA)
- Probability calibration
- Lay-language explanations
- `/api/cxr/predict`

---

### 2️⃣ Clinical Research (SQL + LLM)
- Intent classification:
  - **Statistical** → SQL aggregation
  - **Reasoning** → Case-based retrieval
- SQL templates stored in MongoDB (vector search)
- Auto-tuned SQL using LLMs
- Read-only guards + auto-limits
- `/api/research`

---

### 3️⃣ Case-Based Diagnosis (RAG)
- Vector search over real hospital cases
- Similar-patient retrieval
- Multi-agent reasoning (diagnostician, differential, clinician)
- Evidence-linked outputs (patient IDs)
- JSON-structured results
- `/api/diagnose`

---

## 🧱 Tech Stack

- **Backend:** FastAPI
- **Vision:** TorchXRayVision, PyTorch
- **NLP:** SentenceTransformers, Gemini, Azure OpenAI
- **Databases:**
  - PostgreSQL (MIMIC-IV)
  - MongoDB Atlas (vector search)
- **Safety:** SQL guards, no-write enforcement
- **Deployment:** Uvicorn, Procfile

---

## 📂 Project Structure

```
clinical-ai-assistant-backend/
├── main.py # FastAPI app
├── requirements.txt
├── Procfile
└── README.md
```


(Modular split recommended for production)

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```


## Example Use Cases

Clinical research queries
Hospital data exploration
Medical AI prototyping
Decision-support system demos
AI safety demonstrations in healthcare

