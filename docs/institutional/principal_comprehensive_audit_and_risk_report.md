# CLASSAUDIT AI — INSTITUTIONAL AUDIT & RISK ASSESSMENT REPORT
**Document Reference:** CA-AI-AUDIT-2026-V1  
**Target Authority:** School Principal / Executive Administration  
**System Target:** ClassAudit AI RAG Knowledge Base & Compliance Portal  
**Effective Date:** Academic Session 2026–2027  

---

## 1. EXECUTIVE SUMMARY & INSTITUTIONAL SCOPE

This comprehensive audit and policy document governs the deployment, operational protocols, compliance standards, and risk mitigation strategies for the **ClassAudit AI Smart Face Recognition Attendance & Live Monitoring System**. 

The primary objective of ClassAudit AI is to eliminate manual attendance registers, eradicate proxy attendance, ensure continuous presence of faculty during assigned lecture hours, and maintain transparent automated record-keeping across all academic departments.

Principals and Administrators are required to upload this official document to the **ClassAudit AI Portal Knowledge Base** to enable automated RAG (Retrieval-Augmented Generation) assistant responses for faculty, department heads, and auditing boards.

---

## 2. STANDARD OPERATING PROCEDURES (SOP) FOR FACULTY & ADMIN

### 2.1 Faculty Registration & 5-Photo Embedding Setup
1. **Initial Enrollment:** All newly appointed faculty members must complete face enrollment prior to their first teaching session.
2. **Multi-Angle Capture:** The system captures **5 distinct facial images** across different angles (frontal, 15° left, 15° right, slight upward tilt, and natural expression) to construct a robust 128-dimensional vector embedding.
3. **Threshold Calibration:** Facial matching is strictly validated using a Euclidean Distance threshold of **$\le 0.45$**. Any match score above 0.45 is automatically rejected to guarantee zero cross-identity false matches.

### 2.2 Daily Check-In & Attendance Marking
1. **Morning Check-In Window:** Faculty must complete morning facial check-in via the web/mobile portal between 08:00 AM and 09:00 AM.
2. **Grace Period & Late Marking:**
   - **08:00 AM – 08:45 AM:** Marked as **Present (On-Time)**.
   - **08:46 AM – 09:15 AM:** Marked as **Late Check-in** (Automated administrative flag).
   - **Post 09:15 AM:** Marked as **Absent / Unauthorized Delay** unless pre-approved by the Principal.

---

## 3. LIVE CLASSROOM MONITORING & LECTURE AUDITS

1. **Continuous 5-Second Heartbeat Scan:** During an active lecture session, the class portal executes a periodic 5-second camera scan to verify continuous faculty presence.
2. **Active Teaching Time Accumulation:** Verified seconds are aggregated into the `ClassSession` model (`total_active_duration`).
3. **Auto-Pause & Resumption:** 
   - If the faculty member steps out of camera view for more than 15 seconds, the active duration counter automatically pauses.
   - Upon returning to view, facial recognition re-verifies identity and seamlessly resumes the session timer.
4. **Minimum Lecture Threshold:** A standard 50-minute lecture session requires a minimum of **40 minutes of verified active presence** (80% presence ratio) to be logged as a completed class session.

---

## 4. COMPREHENSIVE 14-POINT RISK ASSESSMENT & MITIGATION FRAMEWORK

To ensure system reliability, privacy compliance, legal security, and institutional integrity, ClassAudit AI incorporates the following 14-point risk analysis and operational mitigation protocol:

### Risk 1: Photo & Video Spoofing (Presentation Attacks)
- **Risk:** An individual attempts to mark fake attendance by presenting a printed photograph, high-resolution tablet image, or pre-recorded video of a registered teacher to the camera.
- **Mitigation:** The system combines multi-angle registration embeddings, strict Euclidean thresholding ($\le 0.45$), and periodic 5-second live micro-motion verification. Static photos fail to sustain dynamic 5-second session heartbeats.

### Risk 2: Low-Illumination & Poor Lighting Environments
- **Risk:** Classrooms with poor natural lighting or faulty light fixtures fail to produce clear facial bounding boxes, causing false rejections.
- **Mitigation:** OpenCV pre-processing applies Histogram Equalization and Adaptive Gamma Correction to adjust frame brightness and contrast before feeding images into the ResNet vector extraction pipeline.

### Risk 3: Network Disruption & Offline Classroom Operations
- **Risk:** Rural schools or campus buildings experiencing temporary internet outages cannot transmit live stream frames to the server.
- **Mitigation:** The client-side application utilizes local IndexedDB queueing to store timestamped, cryptographically signed session verification hashes. Upon network restoration, logs sync automatically with the Django backend.

### Risk 4: False Rejections Due to Facial Alterations (Glasses, Beards, Makeup)
- **Risk:** Faculty changing hairstyles, growing beards, wearing reading glasses, or applying heavy makeup experience high false rejection rates.
- **Mitigation:** The 128-D ResNet deep learning model focuses on deep nodal structures (distance between eyes, nose bridge length, jawline geometry) invariant to facial hair or eyewear. Faculty can re-enroll 5 updated photos at any time via the admin portal.

### Risk 5: Substitute Teacher & Guest Lecturer Assignments
- **Risk:** A regular teacher is on leave, and a guest/substitute teacher conducts the class, leading to automated "Teacher Missing" flags.
- **Mitigation:** The Principal Portal features an **"Assign Substitute"** override tool. Principals map substitute faculty to the specific timetable slot, allowing the substitute's face to verify the class session.

### Risk 6: WebCam Occlusion & Lens Tampering
- **Risk:** Camera lenses placed in classrooms are deliberately blocked, tilted away, covered with tape, or smudged.
- **Mitigation:** The continuous 5-second monitoring service detects persistent "No Face Found" or "Blank Frame" conditions for 3 consecutive minutes, triggering an immediate automated alert to the Principal's dashboard.

### Risk 7: Unauthorized Proxy Check-In via Shared Devices
- **Risk:** A teacher logs into their personal portal on another colleague's laptop or mobile phone to trick the system.
- **Mitigation:** Portal logins require dual-factor verification: valid user credentials AND facial recognition matching the specific logged-in user profile. Credentials alone cannot bypass face verification.

### Risk 8: Biometric Privacy & Data Protection Compliance
- **Risk:** Concerns regarding the storage of raw facial images violating data privacy laws and institutional confidentiality.
- **Mitigation:** ClassAudit AI does not store raw biometrics in vector databases. Facial images are processed to extract mathematical 128-D vector arrays ($[\dots]$). Vectors cannot be reverse-engineered back into original human faces.

### Risk 9: Hardware Failure & Camera Malfunction
- **Risk:** A classroom webcam hardware fails mid-lecture, causing an unintended session interruption.
- **Mitigation:** Teachers can trigger a manual "Hardware Fault Report" button on their portal view, which instantly notifies IT support and pauses the attendance audit penalty for that specific period.

### Risk 10: System Latency & High Server Load During Peak Hours
- **Risk:** Hundreds of teachers logging in simultaneously at 08:00 AM cause server slowdowns or request timeouts.
- **Mitigation:** Django handles request routing efficiently, with vector comparisons offloaded to lightweight C-compiled `dlib` bindings. Session authentication requests complete in under 350 milliseconds.

### Risk 11: Power Outages & Device Battery Depletion
- **Risk:** Unannounced power cuts shut down classroom desktop computers and Wi-Fi routers.
- **Mitigation:** ClassAudit AI is fully responsive and accessible via mobile smartphones running on cellular data (4G/5G). Teachers can host and verify live monitoring via mobile device cameras.

### Risk 12: Defaulter Escalation & Malicious Dispute Claims
- **Risk:** A habitually absent teacher disputes automated defaulter reports, claiming system inaccuracy.
- **Mitigation:** Every attendance decision is backed by immutable audit logs: date, exact timestamp, confidence score, duration counter, and session status log stored in `TeacherAttendance` and `ClassSession` tables.

### Risk 13: Vector Database Corruption & Data Loss
- **Risk:** Server crash or disk failure corrupting the ChromaDB collection or SQLite database.
- **Mitigation:** Automated daily database backups (`db.sqlite3` and `chroma_db/` directory snapshots) are created at 11:59 PM every night and archived to isolated secure storage.

### Risk 14: Model Drift & Algorithm Incompatibility Across Updates
- **Risk:** Software libraries or OS updates degrading recognition accuracy over time.
- **Mitigation:** Strict version pinning in `requirements.txt` (Python 3.14, Django 6.0, dlib, face_recognition) combined with automated regression testing scripts (`generate_defaulters.py` & `tests.py`) before deployment.

---

## 5. PRINCIPAL PORTAL UPLOAD & RAG CHATBOT INGESTION

To ingest this document into the **ClassAudit AI Chatbot**:
1. Log in to the **Principal Admin Portal**.
2. Navigate to **Document Management / RAG Knowledge Base**.
3. Upload this file (`principal_comprehensive_audit_and_risk_report.md`).
4. Click **"Ingest Document"**.
5. Once ingested, the AI Assistant will instantly answer any questions regarding attendance rules, late policies, live monitoring SOPs, and the 14-point risk framework for all institutional stakeholders.

---
*Report Generated & Certified by ClassAudit AI Compliance Engine*
