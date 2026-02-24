# Use an official Python runtime as a lightweight base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# These are required to compile dlib and for OpenCV to work properly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Make the runtime execution scripts executable
RUN chmod +x /app/start.sh

# Expose the port that Gunicorn will run on
EXPOSE 8000

# Start the application using start.sh wrapper script
CMD ["/app/start.sh"]
