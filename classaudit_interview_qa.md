# ClassAudit AI — AI/ML & Data Science Interview Q&A Guide
> All answers are mathematically grounded in your **actual codebase**. Use this guide to prepare for AI, ML, and Data Science interview tracks.

---

## 👁️ SECTION 1: Face Recognition & Computer Vision (dlib / OpenCV)

---

### Q1. How does the ResNet architecture in dlib generate a 128-dimensional embedding from a face? What does each dimension represent?

**Answer:**
* **Pipeline Mechanism:** The input frame is captured via WebRTC, converted from base64, and decoded using `cv2.imdecode()` to a BGR NumPy array. The `face_recognition.face_encodings()` backend (which wraps `dlib`) runs a deep **ResNet-34** network.
* **ResNet Feature Extraction:** The network processes the face image through 34 layers consisting of convolutional layers, pooling, and residual connections. The residual connections bypass blocks of layers, solving the vanishing gradient problem and allowing the model to learn deep feature maps.
* **Dimensional Representation:** The final fully connected layer outputs a dense **128-dimensional float vector**. These dimensions do not represent isolated human features (such as eye color or nose length). Instead, they represent a latent coordinate space optimized using a **triplet loss function**. The network maps faces of the same individual to nearby coordinates and faces of different individuals to distant coordinates in this high-dimensional continuous space.

---

### Q2. Why is Euclidean Distance preferred over Cosine Similarity for evaluating dlib face embeddings? What is the mathematical relationship?

**Answer:**
ED is used because it just match the distance btwn 2 vectors and cS is used when we have to extract the data on the basic of symantic meaningn . and in this case we have to just compare the vectors 



### Q5. How would you architect a lightweight, real-time Liveness Detection (Anti-Spoofing) classifier using OpenCV?

**Answer:**
1- Blink Detection (EAR): Detect natural eye blinking by measuring changes in eye shape across consecutive frames. A real person blinks naturally, while a static photo usually does not.

2- Texture Analysis (LBP): Analyze the texture of the face image. Real skin has natural texture, while printed photos and digital screens may show patterns, reflections, or distortions. An LBP-based SVM classifier can quickly classify the face as real or spoofed.

---

### Q6. How did you compute the ~87% accuracy? What was your test/validation methodology?

**Answer:**
Sir, humne model validate karne ke liye ek simple testing script likhi thi. Humne do folders banaye:

1. **Genuine Folder:** Registered teachers ki 100 photos.
2. **Impostor Folder:** Unregistered/unknown logon ki 100 photos.

Humne Python script ke through dono folders ki images ko project ke main function `match_face()` se run kiya.

- **True Positive (TP = 74):** Genuine folder ki 100 photos mein se **74 baar** face successfully match hua.
- **False Negative (FN = 26):** Baaki **26 photos** poor lighting aur shadows ki wajah se reject ho gayi (False Rejections).
- **True Negative (TN = 100):** Impostor folder ki **100 mein se 100 photos** successfully reject hui (Correct Rejections).
- **False Positive (FP = 0):** Kisi bhi unknown person ko galti se registered user ke roop mein accept nahi kiya gaya (Zero False Acceptances).

### Results

- **Overall Accuracy:** `(TP + TN) / Total Attempts = (74 + 100) / 200 = 87%`
- **False Accept Rate (FAR):** `0%` (Jo target `<0.05%` ko confirm karta hai)
- **Genuine Accept Rate (GAR):** `74%`
---

## 🤖 SECTION 2: Retrieval-Augmented Generation (RAG) & Vector Databases

---

### Q7. Why did you choose `all-MiniLM-L6-v2` as your embedding model? What are its limits?

**Answer:**
- **Lightweight & Fast:** Ye ek lightweight model hai aur text ko **384-dimensional embeddings** mein convert karta hai, jo semantic search ke liye fast aur efficient hai.

1. **Limitation:** Iski input limit around **256 tokens** hai. Agar text chunk bahut bada ho, to remaining text truncate ho sakta hai aur information loss ho sakti hai.

2. **Solution:** Isliye humne custom chunking use ki aur text chunks ko around **600 characters** tak limit kiya, taaki important content truncate na ho.
---

### Q9. Explain your custom paragraph-aware chunking strategy. How do chunk size and overlap impact RAG performance?

**Answer:**
### Q9. Explain your custom paragraph-aware chunking strategy. How do chunk size and overlap impact RAG performance?

**Answer:**

- **Chunk Size:** Hum chunk size ko around **600 characters** tak limit karte hain, taaki embedding model ki token limit cross na ho. Bahut bade chunks information truncate kar sakte hain, aur bahut chhote chunks important context lose kar dete hain.

- **Overlap:** Currently hum **zero overlap** use karte hain taaki chunks distinct rahein. Lekin small overlap, jaise **100 characters**, context ko preserve karne mein help kar sakta hai jab information do chunks ke beech split ho jaye.

- **Noise Removal:** Hum **30 characters se chhote chunks** ko remove kar dete hain kyunki unmein mostly headings ya unnecessary information hoti hai.

---


### Q11. How do you evaluate the quality of your RAG? How would you calculate Faithfulness, Answer Relevance, and Context Recall?

**Answer:**

> "Sir, humne apni RAG pipeline ki quality evaluate karne ke liye **RAGAS framework** use kiya hai. Humne around **20 control questions** aur unke ground-truth answers ka evaluation dataset banaya.
>
> Har query ke liye hum **question, retrieved contexts, generated answer, aur ground truth** collect karke RAGAS evaluation pipeline mein pass karte hain.
>
> Hum mainly 3 metrics evaluate karte hain:
>
> **Faithfulness:** Ye check karta hai ki generated answer retrieved context ke basis par hai ya model ne hallucination ki hai. Higher score means answer context se properly supported hai.
>
> **Answer Relevance:** Ye check karta hai ki generated answer user ke actual question ko directly answer kar raha hai ya nahi.
>
> **Context Recall:** Ye check karta hai ki retriever ne ground-truth answer ke liye required information retrieve ki hai ya nahi.
>
> In metrics ke scores ko average karke hum overall RAG pipeline ki performance evaluate karte hain. Isse hume clearly identify karne mein help milti hai ki issue retrieval side mein hai ya answer generation side mein."

### Quick Remember

- **Faithfulness →** Answer context se supported hai?
- **Answer Relevance →** Answer question ko address kar raha hai?
- **Context Recall →** Required information retrieve hui?

---

### Q12. What is the 14-point risk assessment and how is the teacher risk score calculated?

**Answer:**

* **14-Point Risk Assessment:** Yeh hamare system ka operational risk analysis framework hai jo 14 areas ko handle karta hai (jaise Presentation Attacks/Photo spoofing, low-light, camera occlusion, device proxy check-in, database corruption, network disruption, etc.).
* **Risk Score Kaise Badhta Hai? (Max 10 Points):** Monthly activity logs ke basis par 5 indicators check hote hain:
  1. **Attendance Consistency < 75%** $\to$ $+3$ pts ($75-85\%$ $\to$ $+2$ pts)
  2. **Early Exits > 3 times** $\to$ $+2$ pts
  3. **Late Entries > 3 times** $\to$ $+1$ pt
  4. **Interruption ( stepping out of webcam view)** $\to$ $+2$ pts
  5. **Completion Rate < 80%** $\to$ $+2$ pts

---

### Q13. Which charts/graphs are used in the project, and what do they represent?

**Answer:**

Frontend par dashboard analytics ke liye **Chart.js** use kiya hai jo yeh 5 charts show karta hai:
1. **Attendance Consistency (Bar Chart):** Har teacher ka active teaching time ratio dikhata hai.
2. **Daily Attendance Trend (Line Chart):** Last 14 days ka attendance volume path show karta hai.
3. **Department Presence (Doughnut Chart):** Kis department ke kitne teachers currently present hain.
4. **Department Performance Index (Grouped Bar Chart):** Average consistency aur class completion rate ka comparison.
5. **Department Risk Density (Heatmap/Polar Area):** Kis department mein sabse zyada violations ho rahe hain (shadows/absences/exits).

---

## ⚡ Quick-Fire ML Reference Table

| Parameter / Question | Technical Answer | Mathematical Metric |
|---|---|---|
| **Biometric Distance Metric** | Euclidean ($L_2$) Distance | $d(u,v) = \sqrt{\sum (u_i - v_i)^2}$ |
| **Vector Space Comparison** | Cosine Space Search | $\text{Similarity} = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$ |
| **Embedding Size** | 128 dimensions (dlib) / 384 dimensions (MiniLM) | Float arrays ($1 \times N$) |
| **MiniLM Context Window** | 256 WordPiece tokens | ~600–700 character limit |
| **LLM Inference Optimization** | Low Temperature (0.3) | $\text{Softmax}(Z / T)$ |
| **ChromaDB Index Algorithm** | Hierarchical Navigable Small World (HNSW) | Approximate Nearest Neighbor (ANN) |
| **Multi-Tenancy Scoping** | Isolated Principal Collections | `principal_{id}` metadata filters |
| **Biometric Enrollment Count** | 5 multi-angle enrollment frames | Median embedding computation |
