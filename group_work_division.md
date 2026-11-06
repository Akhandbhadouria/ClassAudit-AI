# 👥 ClassAudit AI — Group Work Division (6 Members)

> **Project**: ClassAudit AI — AI-Powered Teacher Attendance & Live Class Monitoring System
> **Tech Stack**: Django · Face Recognition · dlib · OpenCV · LLaMA 3.3 70B · ChromaDB · Chart.js

---

## 🧑‍💻 Member 1 — Project Overview & Architecture
**Role**: Project Lead / System Architect

### 📌 Topics to Cover
- Project ka introduction aur problem statement
- Kyon yeh system banaya gaya (manual attendance ki problems)
- System ka overall architecture diagram (Principal ↔ Teacher ↔ AI)
- Tech Stack overview (Django, Python, SQLite/PostgreSQL, etc.)
- Project structure ka walkthrough:
  ```
  face-recognition-authentication/
  ├── facere/     → Django config
  ├── accounts/   → Core app
  ├── chatbot/    → RAG Engine
  └── static/     → CSS + JS
  ```
- 6 Database Models explain karna:
  - `Principal`, `Teacher`, `Timetable`, `TeacherAttendance`, `ClassSession`, `PrincipalDocument`

### 📁 Key Files
- [`project_analysis.md`](file:///Users/lambardaar/Downloads/face-recognition-authentication/project_analysis.md)
- [`README.md`](file:///Users/lambardaar/Downloads/face-recognition-authentication/README.md)
- [`facere/`](file:///Users/lambardaar/Downloads/face-recognition-authentication/facere/) — Django settings, URLs, WSGI

### 🎯 Key Stats to Mention
| Metric | Value |
|---|---|
| URL Routes | 27 |
| HTML Templates | 19 |
| CSS Stylesheets | 18 |
| Django Models | 6 |
| Core View Logic | ~1,460 lines |

---

## 🧑‍💻 Member 2 — Authentication System & Role-Based Access
**Role**: Security & Auth Module

### 📌 Topics to Cover
- **Face Recognition Login** kaise kaam karta hai:
  - Webcam se face capture → 128-D embedding → DB se match → login
  - Threshold: `0.45` (Euclidean distance)
- **Password Login** — fallback system
- **Role-Based Access Control (RBAC)**:
  - Principal Portal vs Teacher Portal — alag dashboards, alag URLs
- **Multi-Image Enrollment** during registration:
  - 5 photos capture → embeddings average → DB mein store
- Security benefits: proxy attendance eliminate

### 📁 Key Files
- [`accounts/views.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/) — login, register views
- [`accounts/models.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/) — User, Teacher, Principal models
- Face templates: `login.html`, `register.html`

### 🎯 Demo Points
- Live webcam login flow dikhana
- Wrong face pe rejection dikhana
- Principal vs Teacher login alag-alag dikhana

---

## 🧑‍💻 Member 3 — Face Recognition Engine (AI Core)
**Role**: Computer Vision / AI Engineer

### 📌 Topics to Cover
- **Face Detection**: `face_recognition.face_locations()` se face dhundna
- **128-Dimensional Embeddings**:
  - `dlib` ka deep learning model use karta hai
  - Ek face = 128 numbers ka vector
- **Averaged Embeddings**: 5 images → average → ek robust embedding
- **Face Matching**: Euclidean distance formula se compare
  ```
  distance = ||embedding1 - embedding2||
  if distance < 0.45 → MATCH ✅
  ```
- **Live Verification Loop**:
  - Class ke dauran har 5 seconds mein face verify hota hai
  - Active teaching duration track hoti hai

### 📁 Key Files
- [`accounts/utils/face_detector.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/)
- [`accounts/utils/face_embedding.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/)
- [`accounts/utils/face_matcher.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/)
- [`data/`](file:///Users/lambardaar/Downloads/face-recognition-authentication/data/) — Face embedding data per user

### 🎯 Explain These Concepts
- Why 128-D embeddings? → Mathematical representation of facial features
- Why threshold 0.45? → Balance between accuracy & false positives
- Why average 5 images? → More robust, handles lighting variations

---

## 🧑‍💻 Member 4 — Principal Portal & Analytics Dashboard
**Role**: Admin Portal & Data Analytics

### 📌 Topics to Cover
- **Principal Dashboard**:
  - Department-wise teacher overview
  - Real-time present/absent statistics
- **Teacher Management**: Add, view, delete teachers + webcam face capture
- **Timetable Scheduling**: Weekly schedule (MON–SAT) management
- **Attendance Reports**: Date/month/year wise filterable reports
- **Performance Analysis** (Chart.js):
  - Departmental distribution pie chart
  - Top teachers bar chart
  - Active hours trend line
- **Defaulter Export**: CSV download — filterable by dept/day/month/year
- **Document Upload (RAG)**: PDF/MD/TXT upload for AI knowledge base

### 📁 Key Files
- [`accounts/views.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/) — Principal views section
- [`static/`](file:///Users/lambardaar/Downloads/face-recognition-authentication/static/) — CSS & Chart.js files
- Templates: `principal_dashboard.html`, `teacher_list.html`, `attendance_report.html`

### 🎯 Demo Points
- Dashboard ke charts live dikhana (Chart.js)
- CSV export button click karke file download dikhana
- Teacher add karne ka flow dikhana

---

## 🧑‍💻 Member 5 — Teacher Portal & Live Class Monitoring
**Role**: Teacher Experience & Real-time Monitoring

### 📌 Topics to Cover
- **Teacher Dashboard**:
  - Welcome banner + today's scheduled classes
  - Class history + calendar widget
- **Mark Attendance**: Daily face-based check-in via webcam
  - Auto-status detection: Present / Late
- **Start Class**: Scheduled class session launch from timetable
- **Live Class Monitoring** (Most Important Feature! ⭐):
  - Har **5 seconds** mein webcam se face verify
  - Agar face nahi mila → timer pause
  - Active presence duration track hoti hai
  - Continuous monitoring = real teaching time tracked
- **Previous Records**: Filterable class session history
- **Profile Page**: Monthly attendance stats (Present/Late/Absent)
- **Help Center**: FAQ page

### 📁 Key Files
- [`accounts/views.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/) — Teacher views section
- Templates: `teacher_dashboard.html`, `start_class.html`, `live_monitoring.html`
- [`accounts/utils/face_matcher.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/) — Live verification logic

### 🎯 Demo Points
- Teacher login → Dashboard dikhana
- Attendance mark karna (face recognition live)
- Live class session start karna — 5-second verification loop dikhana
- Agar face frame se bahar gaya toh timer ruk jaata hai — demonstrate karna

---

## 🧑‍💻 Member 6 — AI Chatbot (RAG Engine) & Deployment
**Role**: AI/ML Engineer & DevOps

### 📌 Topics to Cover

#### Part A — RAG Chatbot 🤖
- **RAG kya hota hai?** (Retrieval-Augmented Generation):
  - Documents upload karo → chunks mein todho → embeddings banao → vector DB mein store karo
  - Query aaye → relevant chunks dhundho → LLM ko bhejo → answer generate karo
- **Per-Principal Knowledge Base**: Har school ka apna isolated ChromaDB collection
- **Document Support**: PDF, Markdown, TXT files ingest kar sakta hai
- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers) — semantic similarity
- **LLM**: **LLaMA 3.3 70B** via **Groq API** — fast inference
- **Chunking Strategy**: 600 chars + 100 overlap — better retrieval accuracy
- **Guardrails**: Sirf uploaded documents se jawab — no hallucination
- **Source Attribution**: Answer ke saath document source bhi batata hai

#### Part B — Deployment & Infrastructure 🚀
- **Gunicorn** — production WSGI server
- **WhiteNoise** — static files serving
- **Procfile** — deployment config
- **Environment Variables** (`.env`) — API keys, DB config
- **Requirements**: `requirements.txt` — all dependencies

### 📁 Key Files
- [`chatbot/`](file:///Users/lambardaar/Downloads/face-recognition-authentication/chatbot/) — RAG engine views & URLs
- [`chroma_db/`](file:///Users/lambardaar/Downloads/face-recognition-authentication/chroma_db/) — Persistent vector store
- [`Procfile`](file:///Users/lambardaar/Downloads/face-recognition-authentication/Procfile) — Deployment config
- [`requirements.txt`](file:///Users/lambardaar/Downloads/face-recognition-authentication/requirements.txt)
- [`.env`](file:///Users/lambardaar/Downloads/face-recognition-authentication/.env) — API keys (Groq, etc.)

### 🎯 Demo Points
- Document upload karna (PDF/TXT)
- Chatbot se policy question poochna
- Source attribution dikhana (answer ke neeche document name)
- Groq API ki speed highlight karna

---

## 📋 Summary Table

| Member | Role | Main Area |
|---|---|---|
| **Member 1** | Project Lead / Architect | Overview, Architecture, DB Models |
| **Member 2** | Security & Auth | Login System, RBAC, Enrollment |
| **Member 3** | Computer Vision / AI | Face Detection, Embeddings, Matching |
| **Member 4** | Admin Portal | Principal Dashboard, Analytics, CSV Export |
| **Member 5** | Teacher Experience | Teacher Portal, Live Monitoring (⭐ Star Feature) |
| **Member 6** | AI/ML & DevOps | RAG Chatbot, LLaMA, ChromaDB, Deployment |

---

## 💡 Presentation Tips

> [!TIP]
> **Demo order**: Member 1 → 2 → 3 → 4 → 5 → 6 follow karo taaki story smoothly flow kare

> [!IMPORTANT]
> **Star Feature** — Live Class Monitoring (Member 5) aur RAG Chatbot (Member 6) sabse impressive hain — inhe achhe se demo karo

> [!NOTE]
> Ek system pe project run karke raho taaki live demo ho sake. Member 2 pehle ek teacher account register karega, phir baaki members usko use karenge demo ke liye.
