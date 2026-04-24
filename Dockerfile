FROM node:20-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libreoffice \
        libgl1 \
        libglib2.0-0 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package-lock.json ./frontend/

RUN npm ci \
    && npm --prefix ./frontend ci

COPY . .

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install \
        "docling>=2.55.0" \
        "pymupdf>=1.24.0" \
        numpy \
        opencv-python-headless \
        pillow \
        pymupdf4llm

RUN npm run frontend:build

ENV PATH="/opt/venv/bin:${PATH}"
ENV NODE_ENV=production
ENV FS_EXPLORER_HOST=0.0.0.0
ENV FS_EXPLORER_PORT=8000
ENV FS_EXPLORER_PYTHON_BIN=/opt/venv/bin/python

EXPOSE 8000

CMD ["npm", "start"]
