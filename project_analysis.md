# 🎓 ClassAudit AI — Project Analysis

---

## 📖 Description

**ClassAudit AI** is a full-stack, AI-powered web application built with **Django** that automates teacher attendance tracking and live class monitoring for educational institutions. It combines **real-time face recognition** (powered by `dlib` + `face_recognition` + `OpenCV`) with a modern dual-portal system — one for **Principals** (admins) and one for **Teachers** — delivering a seamless and intelligent academic management experience.

The system goes beyond simple check-in; it continuously monitors a teacher's physical presence throughout an entire class session and even includes an AI-powered **RAG-based chatbot** backed by **LLaMA 3.3 70B** (via Groq) that answers institutional policy questions from uploaded school documents.

---

## 🎯 Objective

> To replace manual attendance registers and passive class monitoring with an intelligent, biometric system that:

1. **Eliminates proxy attendance** — Face recognition ensures only the physically present teacher is logged in.
2. **Tracks active class presence** — Continuous live face verification every 5 seconds measures real teaching duration, not just login time.
3. **Empowers Principals** — A powerful admin portal gives data-driven insights into teacher performance, department analytics, and attendance trends.
4. **Streamlines administration** — Automates schedule management, report generation, and CSV export of defaulters.
5. **Provides an AI knowledge assistant** — A RAG chatbot lets teachers and principals query institutional policies from uploaded PDF/Markdown/TXT documents.

---

## ✨ Features

### 🔐 Authentication System
| Feature | Detail |
|---|---|
| **Face Recognition Login** | Webcam-based login using 128-dimensional face embeddings; threshold `0.45` for match |
| **Password Login** | Traditional username/password fallback for both Principals and Teachers |
| **Role-Based Access Control** | Completely separate portals, dashboards, and URL namespaces per role |
| **Multi-Image Enrollment** | Captures **5 face images** during registration and averages embeddings for accuracy |

---

### 👨‍💼 Principal Portal
| Feature | Detail |
|---|---|
| **Dashboard** | Teacher overview grouped by department; real-time present/absent statistics |
| **Teacher Management** | Add, view, and delete teachers with live webcam face capture during registration |
| **Timetable Scheduling** | Create and manage weekly class schedules (MON–SAT) per teacher |
| **Attendance Reports** | Detailed, filterable reports by date, month, and year per teacher |
| **Performance Analysis** | Visual charts (Chart.js) for departmental distribution, top teachers, and active hours |
| **Defaulter Export** | One-click CSV export of teachers who missed attendance, filterable by dept/day/month/year |
| **Document Upload (RAG)** | Upload PDF, Markdown, or TXT files to build the institution's AI knowledge base |

---

### 👩‍🏫 Teacher Portal
| Feature | Detail |
|---|---|
| **Dashboard** | Welcome banner, today's scheduled classes, class history, and a calendar widget |
| **Profile** | Personal info, monthly attendance stats (Present / Late / Absent), class report history |
| **Mark Attendance** | Daily face-based check-in via webcam; auto-detects status (Present / Late) |
| **Start Class** | Launch a scheduled class session from the personal timetable |
| **Live Class Monitoring** | Real-time face verification every **5 seconds** during a session — tracks active presence duration |
| **Previous Records** | Filterable history of all past class sessions with duration and status |
| **Help Center** | FAQ and platform guidance page |

---

### 🤖 Face Recognition Engine
| Feature | Detail |
|---|---|
| **Face Detection** | `face_recognition.face_locations()` to locate faces in frames |
| **128-D Embeddings** | `dlib`'s deep learning model generates 128-dimensional face vectors |
| **Averaged Embeddings** | 5 captured images are averaged → single robust embedding stored in DB |
| **Face Matching** | Euclidean distance comparison with configurable threshold (`0.45`) |
| **Live Verification** | Continuous verification loop during class sessions, updates active duration |

---

### 🧠 AI Chatbot (RAG Engine)
| Feature | Detail |
|---|---|
| **Per-Principal Knowledge Base** | Each school has its own isolated ChromaDB vector collection |
| **Document Support** | Ingests **PDF**, **Markdown**, and **TXT** files |
| **Embedding Model** | `all-MiniLM-L6-v2` (via `sentence-transformers`) for semantic chunking |
| **LLM** | **LLaMA 3.3 70B** via **Groq API** for answer generation |
| **Chunking Strategy** | Overlapping paragraph-based chunks (600 chars, 100 overlap) for better retrieval |
| **Source Attribution** | Answers reference the exact document they were retrieved from |
| **Guardrails** | Only answers from uploaded documents; no hallucination from outside knowledge |

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | Django 4.2 (Python 3.10/3.11) |
| **Database** | SQLite3 (dev) / PostgreSQL (prod via `psycopg2`) |
| **Face Recognition** | `face_recognition` 1.3.0, `dlib` 19.24.2, `OpenCV` 4.8, `NumPy` 1.26 |
| **AI / LLM** | Groq API → LLaMA 3.3 70B Versatile |
| **Vector Store** | ChromaDB (persistent, per-principal collections) |
| **Embeddings** | `sentence-transformers` → `all-MiniLM-L6-v2` |
| **Frontend** | HTML5, Vanilla CSS (18 modular stylesheets), Vanilla JavaScript |
| **Charts** | Chart.js |
| **Icons / Fonts** | Font Awesome 6.4, Plus Jakarta Sans, Bungee, Nunito, Inter |
| **Production** | Gunicorn + WhiteNoise |

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
├── facere/              → Django project config (settings, URLs, WSGI/ASGI)
├── accounts/            → Core app: views (~1460 lines), models (6), templates (19)
│   └── utils/           → face_detector.py | face_embedding.py | face_matcher.py
├── chatbot/             → RAG engine + chatbot views/URLs
├── static/              → 18 modular CSS files + JS + images
├── chroma_db/           → Persistent ChromaDB vector store
├── data/                → Face embedding data per user
├── media/               → Uploaded documents / face images
└── requirements.txt     → All Python dependencies
```

---

## 📊 Scale & Complexity

| Metric | Value |
|---|---|
| URL Routes | 27 |
| HTML Templates | 19 |
| CSS Stylesheets | 18 |
| Django Models | 6 |
| Core View Logic | ~1,460 lines |
| RAG Engine | ~304 lines |
| Face Utilities | 3 modules |
| Database | SQLite3 (200 KB current) |
