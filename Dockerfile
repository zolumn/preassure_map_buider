FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY main_separated.py /app/main_separated.py
COPY config/config_separated.yaml /app/config/config_separated.yaml
COPY config/geology.clr /app/config/geology.clr

RUN useradd -ms /bin/bash appuser
USER appuser

ENTRYPOINT ["python", "/app/main_separated.py"]
CMD ["--config", "config/config_separated.yaml"]