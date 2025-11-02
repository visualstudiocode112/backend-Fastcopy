FROM python:3.11-slim

# set workdir
WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY . /app

# expose port
EXPOSE 8000

# default env
ENV PYTHONUNBUFFERED=1
ENV UVICORN_WORKERS=1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
