FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 REELDESK_DATA=/data REELDESK_HOST=0.0.0.0 REELDESK_PORT=8080
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY static ./static
RUN mkdir -p /data && chown 10001:10001 /data
USER 10001:10001
EXPOSE 8080
CMD ["python", "-m", "backend.server"]
