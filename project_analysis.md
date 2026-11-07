# 🎓 ClassAudit AI — Project Analysis

---

## 📖 Description

**ClassAudit AI** is a production-grade, full-stack, AI-powered web application built with **Django 5** and **Django Channels (ASGI)** that automates teacher attendance tracking and real-time live classroom monitoring for educational institutions. It combines **zero-`dlib` face recognition** (powered by **Google MediaPipe** + **ArcFace ONNX** 512-D embeddings) with persistent **WebSocket live telemetry**, **3-layer anti-spoofing**, **multi-face presence resolution**, and a modern dual-portal system for **Principals** (admins) and **Teachers**.

The system goes beyond passive check-in; it continuously monitors a teacher's physical presence throughout an entire class session via WebSockets, tolerates classroom crowds (students in camera view), and includes an AI-powered **multi-turn RAG chatbot** backed by **LLaMA 3.3 70B** (via Groq) that answers institutional policy questions directly from school handbooks.

---

## 🎯 Objectives

> To replace manual attendance registers and passive class monitoring with an intelligent, biometric system that:

1. **Eliminates proxy attendance** — 3D facial depth analysis and natural blink dynamics ensure photos, videos, and phone screens are strictly rejected.
2. **Tracks active class presence in real time** — Persistent WebSocket connection streams biometric telemetry every 5 seconds, measuring real teaching duration without HTTP overhead.
3. **Multi-face classroom tolerance** — When students or colleagues walk into the camera frame, the system dynamically locates the teacher instead of aborting the session.
4. **Empowers Principals** — Data-driven insights into faculty performance, department attendance analytics, and automatic CSV export of defaulters.
5. **Provides an AI knowledge assistant** — A multi-turn RAG chatbot lets faculty query institutional rules, leave policies, and procedures from uploaded PDF/Markdown/TXT files.
6. **Scales effortlessly from dev to production** — Decoupled configuration switches seamlessly between SQLite (local) and pooled PostgreSQL (production).

---

## ✨ Features

### 🔐 Biometric Authentication & Anti-Spoofing
| Feature | Detail |
|---|---|
| **Zero-`dlib` Architecture** | Replaced heavy C++ `dlib` with Google MediaPipe and ONNXRuntime (15x faster, ~20ms inference) |
| **512-D ArcFace Vectors** | MobileFaceNet model outputs dense 512-dimensional normalized embeddings |
| **Cosine Distance Metric** | Matches embeddings using Cosine Distance (`1 - np.dot(u, v)`); threshold `0.45` |
| **3D Depth Liveness** | Analyzes 468 facial mesh landmarks along the $Z$-axis (curvature) to reject 2D screen/paper attacks |
| **Screen Glare Detection** | Spectral reflection filtering detects high-intensity digital display backlights |
| **Natural Blink Verification** | Temporal burst analysis over 8 frames (2.0s) checks eye aspect ratio ($EAR$) dynamics |
| **Password Login Fallback** | Traditional username/password authentication for both Principals and Teachers |
| **Role-Based Access Control** | Strictly separated views, decorators (`@principal_required`), and URL namespaces |

---

### 👨‍💼 Principal Portal (Administration Hub)
| Feature | Detail |
|---|---|
| **Dashboard** | Faculty overview grouped by department; real-time present/late/absent statistics |
| **Teacher Management** | Register faculty with live webcam 512-D biometric vector extraction; manage and delete records |
| **Timetable Scheduling** | Create and manage weekly timetable slots (MON–SAT) per teacher |
| **Attendance Reports** | Comprehensive attendance logs filterable by date, month, and year |
| **Performance Analysis** | Interactive Chart.js charts showing departmental distribution, top faculty, and active hours |
| **Defaulter Export** | Instant CSV export of teachers who missed classes or arrived late |
| **Document Upload (RAG)** | Upload institutional handbooks (PDF, Markdown, TXT) to build the institution's knowledge base |

---

### 👩‍🏫 Teacher Portal & Live Classroom Telemetry
| Feature | Detail |
|---|---|
| **Dashboard** | Welcome banner, today's timetable periods, completed class history, and calendar widget |
| **Profile** | Attendance summary (Present / Late / Absent), total active teaching hours, and session stats |
| **Mark Attendance** | Daily facial check-in via webcam with automatic Present/Late timestamping |
| **Start Class** | Launch a scheduled class session from the assigned timetable |
| **Live WebSocket Monitoring** | Bidirectional WebSocket connection (`ws://<host>/ws/attendance/`) streams frames every 5s |
| **Multi-Face Presence Resolution** | Identifies the teacher among multiple people in the room; session duration never breaks |
| **Client-Side Pre-Filtering** | Browser hardware-accelerated `FaceDetector` API skips empty frames, saving bandwidth |
| **Previous Records** | Searchable archive of all past lectures with verified duration and status |
| **Institutional AI Widget** | Floating assistant with multi-turn conversation memory for policy queries |

---

### 🧠 AI Chatbot (RAG Engine with Multi-Turn Memory)
| Feature | Detail |
|---|---|
| **Multi-Tenancy Scoping** | Isolated ChromaDB collections per school (`principal_{id}`) |
| **Document Support** | Ingests **PDF**, **Markdown**, and **TXT** files with automatic sanitization |
| **Embedding Engine** | `sentence-transformers` (`all-MiniLM-L6-v2`) generating 384-D dense embeddings |
| **Contextual Memory** | Maintains rolling session history (last 6 turns / 3 Q&A pairs) for conversational follow-ups |
| **LLM Inference** | **LLaMA 3.3 70B** hosted on **Groq API** with low-latency streaming inference |
| **Chunking Strategy** | Paragraph-aware chunking (600 characters, 100 overlap) |
| **Source Citation** | Every generated response cites the exact document title and section retrieved |
| **Hallucination Guardrails** | Strictly grounded prompt forbids making up non-existent institutional policies |

---

## 🏗️ Modern Tech Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | Python 3.10+, Django 5.1.3 |
| **Asynchronous & WebSockets** | Django Channels 4.1.0, Daphne 4.1.2 (ASGI) |
| **Database** | SQLite3 (dev) / PostgreSQL with Connection Pooling (`CONN_MAX_AGE=60`, prod) |
| **Face Detection & Mesh** | Google MediaPipe 0.10.14 (FaceDetection & 468-point 3D FaceMesh) |
| **Face Recognition Engine** | ArcFace MobileFaceNet ONNX (ONNXRuntime 1.23.2, 512-D vectors) |
| **Face Matching** | Cosine Distance Metric (`1 - np.dot(u, v)`) |
| **Anti-Spoofing & Liveness** | 3D Curvature $Z$-Analysis + Screen Glare Filter + Eye Aspect Ratio ($EAR$) |
| **Vector Database** | ChromaDB 0.5.18 (Persistent, Multi-Tenant per Principal) |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-D) |
| **LLM Inference** | Groq API (`openai/gpt-oss-120b` / LLaMA 3.3 70B) |
| **Frontend** | HTML5, Vanilla CSS (18 modular stylesheets), Vanilla JavaScript |
| **Charts** | Chart.js |
| **Typography / Icons** | Font Awesome 6.4, Plus Jakarta Sans, Nunito, Inter |
| **Production Server** | Daphne ASGI + Gunicorn / WhiteNoise |

---

## 🗃️ Database Models (6 Models)

```
Principal ─┐
           ├──< Teacher ──< Timetable
           │         ├──< TeacherAttendance
           │         └──< ClassSession ──> Timetable
           └──< PrincipalDocument (RAG knowledge base)

User (Django built-in) ──< UserImages (face photos)
```

---

## 📁 Project Structure Summary

```
face-recognition-authentication/
├── facere/              → Project config: settings (DB router & Channels), URLs, ASGI router
├── accounts/            → Core app: views, models, consumers (WS), routing (WS), templates
│   └── utils/           → face_detector.py | face_embedding.py | face_matcher.py | liveness_detector.py
├── chatbot/             → Multi-turn RAG engine, document ingestion, Groq integration, chat views
├── static/              → 18 modular CSS files + JS + brand assets
├── models/              → Auto-downloaded ArcFace ONNX model (mobilefacenet.onnx)
├── chroma_db/           → Persistent ChromaDB vector store per principal
├── data/users/          → Normalized 512-D embedding vectors per user (.npy)
├── media/               → Uploaded documents and teacher profile photos
└── requirements.txt     → Production Python dependencies
```

---

## 📊 Scale & Architectural Metrics

| Metric | Value |
|---|---|
| HTTP URL Routes | 28 |
| WebSocket URL Routes | 1 (`ws://<host>/ws/attendance/`) |
| HTML Templates | 19 |
| CSS Stylesheets | 18 |
| Django Models | 6 |
| Biometric Engine Modules | 4 (`face_detector`, `face_embedding`, `face_matcher`, `liveness_detector`) |
| Biometric Latency | ~20ms per frame (vs ~300ms previously with dlib) |
| RAG Memory Capacity | Rolling 6-turn history (3 complete Q&A interactions) |
| Database Concurrency | SQLite (local dev) / PostgreSQL with Connection Pooling (production) |
