FROM python:3.11-slim

# OpenCV / video codec system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (required by some hosts, e.g. Hugging Face Spaces; harmless
# elsewhere) with a writable home so pip/cache/matplotlib etc. don't fail.
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p static/uploads static/uploads/crops && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser

# Default port 7860 matches Hugging Face Spaces' expected app_port out of
# the box. Render (and most other hosts) inject their own $PORT at runtime,
# which overrides this default automatically via the CMD below.
ENV PORT=7860
# Cap native thread pools (OpenBLAS/MKL/OMP) used by torch/numpy/opencv.
# On a low-CPU/low-RAM instance, multi-threaded math libraries just burn
# extra RAM on thread stacks with no speed benefit and worsen memory pressure.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
EXPOSE 7860

CMD gunicorn app:app --bind 0.0.0.0:${PORT} --timeout 300 --workers 1 --threads 2
