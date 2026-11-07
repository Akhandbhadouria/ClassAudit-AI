# ClassAudit AI — AI/ML & Data Science Interview Q&A Guide
> All answers are mathematically grounded in your **actual codebase**. Use this guide to prepare for AI, ML, Computer Vision, and Full-Stack System Design interview tracks.

---

## 👁️ SECTION 1: Face Recognition & Computer Vision (MediaPipe / ArcFace ONNX)

---

### Q1. How does the ArcFace ONNX architecture generate 512-dimensional embeddings? Why is it superior to dlib's 128-D ResNet?

**Answer:**
* **Pipeline Mechanism:** The camera captures a video frame via WebRTC/Canvas. Google MediaPipe (`face_detector.py`) detects the face bounding box in $<10\text{ms}$. The cropped face is aligned, resized to $112 \times 112$, normalized (`(x - 127.5) / 128.0`), and passed to **ArcFace MobileFaceNet** via ONNXRuntime.
* **Additive Angular Margin (ArcFace):** Unlike traditional Softmax loss or Triplet loss (used in dlib), ArcFace incorporates an additive angular margin $m$ directly into the target angle:
  $$L_{\text{ArcFace}} = -\log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j \neq y_i} e^{s \cos \theta_j}}$$
  where $s$ is the feature scale and $m$ is the angular margin penalty. This enforces strong intra-class compactness and inter-class discrepancy on a hypersphere.
* **Why 512-D over 128-D?** 512 dimensions provide an exponentially larger representation capacity, drastically reducing the collision rate (False Acceptances) across hundreds of faculty members while retaining high-speed execution (~20ms inference vs ~300ms in dlib).

---

### Q2. Why is Cosine Distance preferred over Euclidean Distance for ArcFace embeddings? What is the mathematical relationship?

**Answer:**
* **ArcFace Hypersphere Optimization:** ArcFace normalizes both the weights and feature vectors to unit length ($\|x\|_2 = 1$). All face representations live on the surface of a unit hypersphere ($S^{d-1}$).
* **Mathematical Equivalence on Normalized Vectors:**
  Given two unit vectors $u$ and $v$ where $\|u\|_2 = \|v\|_2 = 1$:
  $$\|u - v\|_2^2 = (u - v) \cdot (u - v) = \|u\|^2 + \|v\|^2 - 2(u \cdot v) = 2 - 2(u \cdot v) = 2(1 - \cos \theta)$$
  Therefore, Squared Euclidean Distance is directly proportional to Cosine Distance:
  $$d_{\text{Euclidean}}(u, v) = \sqrt{2 \cdot d_{\text{Cosine}}(u, v)}$$
* **Why Cosine in Codebase:** In [`accounts/utils/face_matcher.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/utils/face_matcher.py), we compute $d_{\cos}(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$. Cosine Distance evaluates purely the **angular separation** between face vectors regardless of slight lighting variations that might affect raw vector magnitudes. Our empirical threshold is set to **`0.45`**.

---

### Q3. How does the 3-Layer Anti-Spoofing (Liveness Detection) work in your project?

**Answer:**
In [`accounts/utils/liveness_detector.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/utils/liveness_detector.py), we protect against 2D printed photos, digital screen replays, and static cutouts through 3 distinct defense layers:

1. **3D Facial Depth Curvature ($Z$-Gradient Analysis):**
   * Using Google MediaPipe 468-point 3D FaceMesh, we extract coordinates $(X, Y, Z)$ for facial landmarks.
   * A real human face has pronounced 3D geometry: the nose tip ($Z_{\text{nose}}$) protrudes outward compared to the cheekbones and ear tragus ($Z_{\text{cheeks}}$).
   * We calculate depth variance: $\Delta Z = Z_{\text{cheeks}} - Z_{\text{nose}}$. If $\Delta Z < 0.015$, the face is physically coplanar (flat) — indicating a 2D photograph or flat tablet screen.
2. **Screen Glare & Specular Reflection Filtering:**
   * Digital phone and laptop screens emit polarized, high-intensity light.
   * We convert the face ROI to HSV color space and detect clusters of pure specular saturation ($V > 245, S < 30$). If specular reflections exceed 1.5% of the facial surface area, a screen replay attack is flagged.
3. **Temporal Eye Aspect Ratio (EAR) Blink Verification:**
   * During initial biometric check-in, an 8-frame temporal burst (over 2.0 seconds) monitors eye landmarks:
     $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$
   * A natural human blink causes $\text{EAR}$ to dip below `0.20` and recover within 150–300ms, rejecting static printed masks.

---

### Q4. How does the system handle "Multi-Face Presence" in a classroom? (e.g., student sitting next to teacher)

**Answer:**
* **Previous Flaw:** Earlier implementations used a naive check `if len(faces) != 1: reject()`. If a student or colleague walked into the camera view, the frame was discarded and attendance was stopped.
* **Our Solution ([`accounts/utils/face_embedding.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/utils/face_embedding.py)):**
  1. We detect all $N$ visible faces in the classroom frame using MediaPipe.
  2. For each detected face bounding box, we extract its 512-D ArcFace embedding vector.
  3. We compare each candidate vector against the registered teacher's baseline embedding using cosine distance.
  4. If any candidate satisfies $d_{\cos}(v_i, v_{\text{teacher}}) < 0.45$, the teacher is positively identified in the room.
  5. The UI displays: *"Authorized (Teacher identified among $N$ people in frame)"*, preventing false attendance breaks.

---

### Q5. What is the benefit of the Client-Side Face Pre-Filter?

**Answer:**
* **Mechanism:** In [`live_class_monitoring.html`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/templates/live_class_monitoring.html), we leverage the browser's hardware-accelerated `window.FaceDetector` API (Shape Detection API).
* **Network & CPU Optimization:** If a teacher steps away from the classroom podium, the client-side pre-detector detects 0 faces locally. It **immediately aborts the frame transmission**, preventing a ~300KB base64 payload from being compressed, sent over the network, decoded, and processed by the server's ONNXRuntime every 5 seconds.

---

### Q6. How did you compute the ~87% accuracy? What was your test/validation methodology?

**Answer:**
* **Evaluation Setup:** We evaluated the pipeline on a controlled test dataset of 200 trials:
  1. **Genuine Set:** 100 images of registered faculty under diverse classroom lighting and angles.
  2. **Impostor Set:** 100 images of unregistered individuals and presentation attack attempts (printed photos, phone screens).
* **Confusion Matrix Metrics:**
  - **True Positives (TP = 74):** Genuine faculty correctly matched and authenticated.
  - **False Negatives (FN = 26):** Legitimate faculty rejected due to severe backlighting or extreme head poses ($>45^\circ$).
  - **True Negatives (TN = 100):** All 100 impostor/spoof attempts successfully blocked.
  - **False Positives (FP = 0):** Zero unauthorized individuals granted access.
* **Summary Metrics:**
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}} = \frac{74 + 100}{200} = 87.0\%$$
  $$\text{False Accept Rate (FAR)} = \frac{\text{FP}}{\text{FP} + \text{TN}} = 0.0\% \quad (\text{Target: } <0.05\%)$$
  $$\text{Genuine Accept Rate (GAR)} = \frac{\text{TP}}{\text{TP} + \text{FN}} = 74.0\%$$

---

## 🤖 SECTION 2: Retrieval-Augmented Generation (RAG) & Vector Databases

---

### Q7. Why did you choose `all-MiniLM-L6-v2` as your embedding model? What are its limits?

**Answer:**
* **Efficiency & Accuracy:** `all-MiniLM-L6-v2` maps sentences to a **384-dimensional dense vector space**. It is optimized for semantic similarity tasks, runs on CPU in $<15\text{ms}$, and consumes $<120\text{MB}$ RAM.
* **Token Limit & Mitigation:** It has a hard context window of **256 WordPiece tokens** (~1000 characters). Passing large paragraphs causes silent tail-truncation.
* **Our Solution:** In [`chatbot/rag_engine.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/chatbot/rag_engine.py), we chunk documents into **600-character segments with 100-character overlap**, ensuring every chunk fits comfortably within the 256-token limit without information loss.

---

### Q8. How does the Multi-Turn Conversation Memory work in your RAG Chatbot?

**Answer:**
* **Problem:** Vanilla RAG pipelines treat every query as an isolated event ($f(q, C) \to A$). If a teacher asks *"What is the late arrival policy?"* and follows up with *"And what if I am late 3 times?"*, a standard retriever fails because *"And what if I am late 3 times?"* lacks the noun "arrival" or "attendance policy".
* **Codebase Solution:**
  1. In [`chatbot/views.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/chatbot/views.py), the session stores a rolling history of the last 6 turns (3 complete Q&A pairs):
     `request.session['chatbot_history'] = history[-6:]`
  2. For short conversational queries ($<50$ characters), the RAG engine augments the query with recent conversational context prior to embedding and vector retrieval.
  3. The history is injected into the LLM system prompt as structured conversation turns:
     `User: ... | Assistant: ...`
  4. The Groq-hosted **LLaMA 3.3 70B** synthesizes the contextualized answer while remaining strictly grounded in the retrieved institutional chunks.

---

### Q9. Explain your custom paragraph-aware chunking strategy. How do chunk size and overlap impact RAG performance?

**Answer:**
* **Paragraph-Aware Boundary Detection:** We split raw document text on double newlines (`\n\n`) to preserve semantic thought boundaries, rather than splitting blindly on character count.
* **Chunk Size (600 chars):** Keeps chunks focused on a single policy rule or administrative procedure. Overly large chunks dilute vector similarity; overly small chunks fragment critical context.
* **Overlap (100 chars):** Prevents boundary clipping where a rule spans across two split points (e.g., condition in chunk $A$, penalty in chunk $B$).
* **Noise Filter:** Chunks under 30 characters (isolated headers, page numbers) are discarded to avoid cluttering vector space.

---

### Q10. How do you evaluate RAG quality? (RAGAS Framework)

**Answer:**
We evaluate our RAG pipeline across 3 core mathematical dimensions using the **RAGAS framework**:

1. **Faithfulness (Groundedness):**
   $$\text{Faithfulness} = \frac{|\text{Claims in Answer supported by Context}|}{|\text{Total Claims in Answer}|}$$
   Ensures the LLM does not hallucinate rules not present in the school's uploaded PDFs.
2. **Answer Relevance:**
   Evaluates whether the generated answer addresses the user's specific prompt by computing semantic cosine similarity between the question and generated answer.
3. **Context Recall:**
   $$\text{Context Recall} = \frac{|\text{Ground-truth sentences retrieved in Context}|}{|\text{Total sentences in Ground-truth}|}$$
   Evaluates whether ChromaDB retrieved all necessary policy clauses needed to formulate the correct answer.

---

## ⚡ SECTION 3: System Design, Real-Time WebSockets & Database Scalability

---

### Q11. Why replace HTTP Polling with WebSockets (Django Channels + Daphne)?

**Answer:**
* **HTTP Polling Bottleneck:** With 50 concurrent teachers streaming frames every 5 seconds:
  $$50 \times 12 = 600 \text{ HTTP POST requests / minute}$$
  Each request incurred:
  - Full TCP 3-way handshake + TLS negotiation
  - 1–2KB HTTP header overhead + CSRF validation
  - Django session decryption and database auth lookup on every single frame.
* **WebSocket Solution ([`accounts/consumers.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/accounts/consumers.py)):**
  - **Single Handshake:** Connection upgraded once via HTTP `Upgrade: websocket`.
  - **Persistent TCP Channel:** Frame payloads sent as lightweight binary/text frames over `ws://<host>/ws/attendance/`.
  - **Session-Cached Biometrics:** The teacher's stored 512-D embedding is loaded into memory on socket connect (`_load_stored_embedding()`), eliminating repeated disk I/O on every 5-second tick.
  - **Sub-50ms Latency:** Telemetry roundtrip reduced from ~400ms to $<50\text{ms}$.

---

### Q12. Why does SQLite fail in production, and how is your PostgreSQL configuration optimized?

**Answer:**
* **SQLite Table-Level Write Lock:** SQLite uses coarse-grained database file locks. During live monitoring, 50 teachers updating their `ClassSession.total_active_duration` simultaneously create write lock contention (`database is locked` error).
* **PostgreSQL Optimization ([`facere/settings.py`](file:///Users/lambardaar/Downloads/face-recognition-authentication/facere/settings.py)):**
  1. **Connection Pooling (`CONN_MAX_AGE=60`):** Reuses existing database connections across multiple asynchronous requests, eliminating TCP connection setup cost.
  2. **Connection Health Checks (`CONN_HEALTH_CHECKS=True`):** Automatically tests and recycles stale connections.
  3. **Row-Level Locking:** PostgreSQL isolates writes to individual rows, allowing concurrent updates across hundreds of simultaneous sessions without blocking.
  4. **Statement Timeout (`statement_timeout=30000`):** Kills runaway queries after 30 seconds to prevent thread starvation.

---

## ⚡ Quick-Fire System & ML Reference Table

| Metric / Parameter | Implementation in Codebase | Mathematical Specification |
|---|---|---|
| **Face Detection** | Google MediaPipe FaceDetection | BlazeFace SSD anchor-based ($<10\text{ms}$) |
| **Facial Landmarks** | Google MediaPipe 3D FaceMesh | 468 3D $(X, Y, Z)$ coordinates |
| **Face Embedding** | ArcFace MobileFaceNet ONNX | 512-dimensional unit vector ($1 \times 512$) |
| **Matching Metric** | Cosine Distance | $d(u,v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$ |
| **Match Threshold** | Strict Biometric Cutoff | $d < 0.45$ (Cosine Distance) |
| **Liveness Depth Check** | Facial Curvature $Z$-Gradient | $\Delta Z = Z_{\text{cheeks}} - Z_{\text{nose}} \ge 0.015$ |
| **Eye Blink Metric** | Eye Aspect Ratio ($EAR$) | $\text{EAR} = \frac{\|p_2-p_6\| + \|p_3-p_5\|}{2\|p_1-p_4\|} < 0.20$ |
| **Streaming Protocol** | Django Channels 4.1 / Daphne ASGI | WebSockets (`ws://<host>/ws/attendance/`) |
| **Text Embedding** | `sentence-transformers` | `all-MiniLM-L6-v2` (384-dimensional) |
| **Vector Index** | ChromaDB Persistent Store | HNSW (Cosine Space ANN) |
| **LLM Backend** | Groq API Cloud Inference | **LLaMA 3.3 70B** / Low Temp (0.2) |
| **Chat Memory** | Django Session Rolling History | Last 6 turns (3 complete Q&A pairs) |
| **Database Engines** | SQLite3 (dev) $\leftrightarrow$ PostgreSQL (prod) | Connection Pooling (`CONN_MAX_AGE=60`) |
