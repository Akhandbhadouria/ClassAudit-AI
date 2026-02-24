# ── Stage 1: Build dlib and other compiled dependencies ──
FROM python:3.10-slim AS builder

# Install system-level build dependencies for dlib/face_recognition
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies (dlib will compile from source here)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Production image ──
FROM python:3.10-slim

# Install only runtime dependencies (no build tools needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libx11-6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Copy and set the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose port (Render sets $PORT automatically)
EXPOSE 10000

# Run migrations and start gunicorn
ENTRYPOINT ["./entrypoint.sh"]
